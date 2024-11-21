#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(portable_simd)]
#![feature(integer_sign_cast)]
#![feature(file_lock)]

mod shuffle_pattern;

use crate::shuffle_pattern::{PATTERNS, PATTERNS_ID};
use std::fs::File;
use std::mem::transmute;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Mask, Simd};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::io;

/// A set to keep track of unique IPs using a bitmask.
struct MaxBitSet {
    words: Vec<AtomicU64>,
}

impl MaxBitSet {
    fn new() -> Self {
        let words = vec![0; 1 << 26].into_iter().map(AtomicU64::new).collect();
        Self { words }
    }

    /// Sets the bit at the specified index.
    fn set(&self, bit_index: u32) {
        let word = unsafe {
            &*self
                .words
                .as_ptr()
                .offset(bit_index.wrapping_shr(6) as isize)
        };
        let mask = 1u64.wrapping_shl(bit_index);
        AtomicU64::fetch_or(word, mask, Ordering::SeqCst);
    }

    /// Returns the number of unique bits set.
    fn count(&self) -> u64 {
        self.words
            .iter()
            .map(|word| word.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }
}

fn read_simd(bytes: &[u8]) -> Simd<u8, 16> {
    if std::intrinsics::likely(bytes.len() >= 16) {
        unsafe { Simd::load_select_ptr(bytes.as_ptr(), Mask::splat(true), Simd::splat(1)) }
    } else {
        let wide = {
            #[cfg(target_endian = "big")]
            {
                let mut wide = 0u128;
                let mut offset = 128 - 8;
                for &b in bytes.iter().take(16) {
                    wide |= (b as u128) << offset;
                    offset -= 8;
                }
                wide
            }
            #[cfg(not(target_endian = "big"))]
            {
                let mut wide = 0u128;
                for &b in bytes.iter().take(16) {
                    wide = (wide << 8) | (b as u128);
                }
                wide
            }
        };
        Simd::from_array(wide.to_ne_bytes())
    }
}

fn newline_location(word: Simd<u8, 16>) -> u32 {
    word.simd_eq(Simd::splat(b'\n'))
        .to_bitmask()
        .trailing_zeros()
}

fn worker(memory: &'static [u8], chunk_id: &'static AtomicUsize, bitset: &'static MaxBitSet) {
    const CHUNK_SIZE: usize = 1024 * 1024;
    loop {
        let id = chunk_id.fetch_add(1, Ordering::SeqCst);
        if std::intrinsics::unlikely(id * CHUNK_SIZE >= memory.len()) {
            break;
        }
        let mut current_mem = &memory[(id * CHUNK_SIZE)..];
        let end = memory[std::cmp::min((id + 1) * CHUNK_SIZE, memory.len())..].as_ptr();

        // Adjusting start
        if std::intrinsics::likely(id > 0) {
            let value = read_simd(current_mem);
            current_mem = &current_mem[(newline_location(value) + 1) as usize..];
        }

        parse_loop(bitset, current_mem, end);
    }
}

fn parse_loop(bitset: &MaxBitSet, mut current_mem: &[u8], end: *const u8) {
    while current_mem.as_ptr() < end {
        let value = read_simd(current_mem);
        let next_new = newline_location(value);
        bitset.set(num_simd(value, next_new));
        current_mem = &current_mem[((next_new + 1) as usize)..];
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: unique_ip_counter <file_path>");
        return Ok(());
    }
    let file_path = &args[1];

    let file = File::open(file_path)?;
    file.lock()?;

    let count = count_ips(&file);

    println!("Number of unique IP addresses: {}", count);
    Ok(())
}

fn count_ips(file: &File) -> u64 {
    // Create memory-mapped file
    // Safety: file is locked, so there should not be any UB
    let mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(file).unwrap() }));

    let chunk_id = AtomicUsize::new(0);
    let bitset = MaxBitSet::new();

    // leaking these items to make it easy to work with threads
    // Safety: all this items live till the end of the program
    let map: &'static [u8] = unsafe { transmute::<&[u8], &'static [u8]>(mmap.as_ref()) };
    let chunk_id: &'static AtomicUsize = Box::leak(Box::new(chunk_id));
    let bitset: &'static MaxBitSet = Box::leak(Box::new(bitset));

    let mut handles = Vec::new();
    let num_threads = num_cpus::get();
    for _ in 0..num_threads {
        handles.push(thread::spawn(move || {
            #[cfg(feature = "max_thread_priority")]
            thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max).unwrap();
            worker(map, chunk_id, bitset);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    bitset.count()
}

/// this is a modified implementation of this V7 algorithm from
/// http://0x80.pl/notesen/2023-04-09-faster-parse-ipv4.html
/// with removed validations and some simplifications
fn num_simd(value: Simd<u8, 16>, new_line: u32) -> u32 {
    let mask = load_mask(new_line);
    let value = mask.select(value, Simd::splat(0));
    let hashcode = calculate_hash(new_line, value);

    let pattern_id = PATTERNS_ID[hashcode as usize];
    let pattern = &PATTERNS[pattern_id as usize];

    let pattern = safe_arch::load_unaligned_m128i(pattern);

    let input = safe_arch::m128i::from(value.to_array());
    let ascii0 = safe_arch::set_splat_i8_m128i('0' as i8);

    let grouped_units_as_chars = safe_arch::shuffle_av_i8z_all_m128i(input, pattern);
    let grouped_units = safe_arch::sub_saturating_u8_m128i(grouped_units_as_chars, ascii0);
    let weights = safe_arch::set_reversed_i8_m128i(
        10, 1, 10, 1, 10, 1, 10, 1, 100, 0, 100, 0, 100, 0, 100, 0,
    );
    let units_and_tenths_combined = safe_arch::mul_u8i8_add_horizontal_saturating_m128i(grouped_units, weights);
    let shifted = safe_arch::combined_byte_shr_imm_m128i::<8>(units_and_tenths_combined, units_and_tenths_combined);
    let combined = safe_arch::add_i16_m128i(shifted, units_and_tenths_combined);
    let packed = safe_arch::pack_i16_to_u8_m128i(combined, combined);
    safe_arch::get_i32_from_m128i_s(packed).cast_unsigned()
}

fn load_mask(new_line: u32) -> Mask<i8, 16> {
    Mask::from_bitmask((1u64 << (new_line)).wrapping_sub(1))
}

#[inline]
fn calculate_hash(new_line: u32, value: Simd<u8, 16>) -> u32 {
    let dot_mask = value.simd_eq(Simd::splat(b'.'));
    let dot_mask_num = dot_mask.to_bitmask() | (1u64 << new_line);
    ((dot_mask_num >> 5) ^ (dot_mask_num & 0x03ff)) as u32
}

#[cfg(test)]
mod tests {
    use std::fs;
    use crate::{newline_location, num_simd, read_simd};
    use rand::Rng;
    use std::simd::Simd;

    /// This is not a test per se
    /// just a "script" to generate test files to be run on. 
    #[test]
    fn generate_test_file() {
        use rand::Rng;
        use std::fs::File;
        use std::io::{self, Write};
        use std::io::BufWriter;

        /// Generates `n` random IP addresses and writes them to the specified file.
        fn generate_ips_and_write_to_file(filename: &str, n: usize) -> io::Result<()> {
            #[cfg(not(debug_assertions))]
            if n >= 100000000 {
                panic!("rerun this test with release mod, otherwise it will take too much time");
            }
            let mut file = BufWriter::new(File::create(filename)?);
            let mut rng = rand::thread_rng();

            for _ in 0..n {
                // Generate a random IP address by creating four random octets.
                let ip = format!(
                    "{}.{}.{}.{}",
                    rng.gen::<u8>(),
                    rng.gen::<u8>(),
                    rng.gen::<u8>(),
                    rng.gen::<u8>()
                );

                writeln!(file, "{}", ip)?;
            }

            Ok(())
        }

        let sizes = [100, 1000, 1000000, 100000000, 1000000000];
        for size in sizes {
            let filename = format!("target/ips-{size}.txt");
            if !fs::metadata(&filename).is_ok() {
                generate_ips_and_write_to_file(&filename, size).unwrap();
            }
        }
    }

    fn ip_to_index(ip: &[u8]) -> Option<u32> {
        let parts: Vec<&[u8]> = ip.split(|&c| c == b'.').collect();
        if parts.len() == 4 {
            let octets: Vec<u8> = parts
                .iter()
                .filter_map(|s| String::from_utf8_lossy(s).parse().ok())
                .collect();
            octets_to_u32(octets)
        } else {
            None
        }
    }

    fn ip_to_index_str(ip: &str) -> Option<u32> {
        let parts: Vec<&str> = ip.split('.').collect();
        if parts.len() == 4 {
            let octets: Vec<u8> = parts.iter().filter_map(|s| s.parse().ok()).collect();
            octets_to_u32(octets)
        } else {
            None
        }
    }

    fn octets_to_u32(octets: Vec<u8>) -> Option<u32> {
        if octets.len() == 4 {
            Some(
                (octets[0] as u32)
                    | ((octets[1] as u32) << 8)
                    | ((octets[2] as u32) << 16)
                    | ((octets[3] as u32) << 24),
            )
        } else {
            None
        }
    }

    #[test]
    fn location_test() {
        let location = newline_location(Simd::from_slice(b"1.1.12.12123.34\n"));
        assert_eq!(location, 15);
        let location = newline_location(Simd::from_slice(b"\n1.1.12.12123.34"));
        assert_eq!(location, 0);
        let location = newline_location(Simd::from_slice(b"1.1.12.12\n123.34"));
        assert_eq!(location, 9);
        let location = newline_location(Simd::from_slice(b"1.1.1.12\n123.34."));
        assert_eq!(location, 8);
        let location = newline_location(Simd::from_slice(b"\n123.234.123.123"));
        assert_eq!(location, 0);
        let location = newline_location(Simd::from_slice(b"1.13.123.123\n123"));
        assert_eq!(location, 12);
        let location = newline_location(Simd::from_slice(b"123.123.123.123\n"));
        assert_eq!(location, 15);
        let location = newline_location(Simd::from_slice(b"1.1.1.1\n1.1.1.1\n"));
        assert_eq!(location, 7);
        let location = newline_location(Simd::from_slice(b"163.162.45.85\n1."));
        assert_eq!(location, 13);
        let location = newline_location(Simd::from_slice(b"163.162.45.8521."));
        assert!(location >= 15);
    }

    #[test]
    fn parse_test_1() {
        let data = Simd::from_array(*b"163.162.45.85\n1.");
        let location = newline_location(data);
        let number = num_simd(data, location);
        let number_real = ip_to_index(&data[..(location as usize)]).unwrap();
        assert_eq!(number, number_real);
    }

    #[test]
    fn parse_test_2() {
        let data = Simd::from_array(*b"1.1.1.1\n2.2.2.2\n");
        let location = newline_location(data);
        let number = num_simd(data, location);
        assert_eq!(number, 0x01010101u32);
    }

    #[test]
    fn parse_test_3() {
        let data = Simd::from_array(*b"1.2.3.4\n2.2.2.2\n");
        let location = newline_location(data);
        let number = num_simd(data, location);
        assert_eq!(number, 0x04030201u32);
    }

    #[test]
    fn simd_x_x_x_x_test() {
        let mut rng = rand::thread_rng();
        let mut ip_str = move || {
            format!(
                "{}.{}.{}.{}",
                rng.gen_range(0..10),
                rng.gen_range(0..10),
                rng.gen_range(0..10),
                rng.gen_range(0..10),
            )
        };
        for _ in 0..1000 {
            let ip1 = ip_str();
            let ip2 = ip_str();
            let data = format!("{ip1}\n{ip2}\n");
            let data_bytes = data.as_bytes()[0..16].try_into().unwrap();
            let location = newline_location(Simd::from_array(data_bytes));
            let parsed = num_simd(read_simd(&data_bytes), location);
            let parsed_correct = ip_to_index_str(&ip1).unwrap();
            assert_eq!(format!("{parsed:#x}"), format!("{parsed_correct:#x}"));
        }
    }

    #[test]
    fn simd_double_digit_test() {
        let mut rng = rand::thread_rng();
        let mut ip_str = move || {
            format!(
                "{}.{}.{}.{}",
                rng.gen_range(0..100),
                rng.gen_range(0..100),
                rng.gen_range(0..100),
                rng.gen_range(0..100),
            )
        };
        for _ in 0..1000 {
            let ip1 = ip_str();
            let ip2 = ip_str();
            let data = format!("{ip1}\n{ip2}\n");
            let data_bytes = data.as_bytes()[0..16].try_into().unwrap();
            let location = newline_location(Simd::from_array(data_bytes));
            let parsed = num_simd(read_simd(&data_bytes), location);
            let parsed_correct = ip_to_index_str(&ip1).unwrap();
            assert_eq!(format!("{parsed:#x}"), format!("{parsed_correct:#x}"));
        }
    }

    #[test]
    fn simd_all_test() {
        let mut rng = rand::thread_rng();
        let mut ip_str = move || {
            format!(
                "{}.{}.{}.{}",
                rng.gen::<u8>(),
                rng.gen::<u8>(),
                rng.gen::<u8>(),
                rng.gen::<u8>(),
            )
        };
        for _ in 0..1000 {
            let ip1 = ip_str();
            let ip2 = ip_str();
            let data = format!("{ip1}\n{ip2}\n");
            let data_bytes = data.as_bytes()[0..16].try_into().unwrap();
            let location = newline_location(Simd::from_array(data_bytes));
            let parsed = num_simd(read_simd(&data_bytes), location);
            let parsed_correct = ip_to_index_str(&ip1).unwrap();
            assert_eq!(
                format!("{parsed:#x}"),
                format!("{parsed_correct:#x}"),
                "parsed {ip1} wrongly"
            );
        }
    }

    #[test]
    fn parse_fuzzy_test() {
        let mut rng = rand::thread_rng();
        let mut ip_str = move || {
            format!(
                "{}.{}.{}.{}",
                rng.gen::<u8>(),
                rng.gen::<u8>(),
                rng.gen::<u8>(),
                rng.gen::<u8>()
            )
        };
        for _ in 0..100000 {
            let ip1 = ip_str();
            let ip2 = ip_str();
            let data = format!("{ip1}\n{ip2}\n");
            let simd_vector = Simd::from_array(data.as_bytes()[0..16].try_into().unwrap());
            let location = newline_location(simd_vector);
            assert_eq!(
                ip1.len(),
                location as usize,
                "length mismatch {ip1:?} {location}"
            );
            let number = num_simd(simd_vector, location);
            let number_real = ip_to_index_str(&ip1).unwrap();
            assert_eq!(number, number_real, "parsing this chunk failed '{data:?}'");
        }
    }
}
