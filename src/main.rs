#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(portable_simd)]
#![feature(integer_sign_cast)]

mod generated_hash;

// use packed_simd::{u8x16, Cast};
use crate::generated_hash::{PATTERNS, PATTERNS_ID};
use std::fs::File;
use std::mem::transmute;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Simd, ToBytes};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::{io, simd};

/// A set to keep track of unique IPs using a bitmask.
struct MaxBitSet {
    // words: Box<[AtomicU64; 1 << 26]>,
    words: Vec<AtomicU64>,
}

impl MaxBitSet {
    fn new() -> Self {
        // let words = Box::new([const { AtomicU64::new(0) }; 1 << 26]);
        let words = vec![0; 1 << 26].into_iter().map(AtomicU64::new).collect();
        Self { words }
    }

    // // #[inline(never)]
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

    // #[inline(never)]
    /// Returns the number of unique bits set.
    fn count(&self) -> u64 {
        self.words
            .iter()
            .map(|word| word.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }
}

// #[inline(never)]
fn num(mut value: u128, new_line: u32) -> u32 {
    value = value & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
    let mut shift_size = 24;
    let mut current_byte: u8 = 0;
    let mut result = 0;

    for _ in 0..new_line {
        let c = (value & 0xFF) as u8;
        value >>= 8;

        if c == 0x0e {
            result |= (current_byte as u32) << shift_size;
            shift_size -= 8;
            current_byte = 0;
        } else {
            current_byte = current_byte.wrapping_mul(10).wrapping_add(c);
        }
    }

    result | (current_byte as u32)
}

// #[inline(never)]
fn num_simd(mut value: u128, new_line: u32) -> u32 {
    unsafe {
        let mask = simd::Mask::from_bitmask((1u64 << new_line) - 1);
        let value = Simd::load_select_or_default(&value.to_be_bytes(), mask);
        let dot_mask = value.simd_eq(Simd::splat('.' as u8));
        let dot_mask_num = dot_mask.to_bitmask() | (1u64 << new_line);
        let hashcode = ((dot_mask_num >> 5) ^ (dot_mask_num & 0x03ff)) as u32;

        let pattern_id = PATTERNS_ID[hashcode as usize];
        let pattern = PATTERNS[pattern_id as usize];
        let max_length = pattern[16];

        use std::arch::x86_64;

        let pattern = x86_64::_mm_loadu_si128(pattern.as_ptr() as *const x86_64::__m128i);

        let input = x86_64::__m128i::from(value);
        let ascii0 = x86_64::_mm_set1_epi8('0' as i8);
        match max_length {
            1 => {
                let t1 = x86_64::_mm_sub_epi8(input, ascii0);
                let t2 = x86_64::_mm_shuffle_epi8(t1, pattern);
                let ipv4 = x86_64::_mm_cvtsi128_si32(t2);
                ipv4.cast_unsigned()
            }
            2 => {
                let t1 = x86_64::_mm_shuffle_epi8(input, pattern);
                let ascii = x86_64::_mm_cvtsi128_si64(t1);
                let w01 = ascii & 0x0f0f0f0f0f0f0f0f;
                let w0 = w01 >> 32;
                let w1 = w01 & 0xffffffffi64;
                ((10 * w1 + w0) as i32).cast_unsigned()
            }
            3 => {
                let t1 = x86_64::_mm_shuffle_epi8(input, pattern);
                let t2 = x86_64::_mm_subs_epu8(t1, ascii0);
                let weights = x86_64::_mm_setr_epi8(
                    10, 1, 10, 1, 10, 1, 10, 1, 100, 0, 100, 0, 100, 0, 100, 0,
                );
                let t3 = x86_64::_mm_maddubs_epi16(t2, weights);
                let t4 = x86_64::_mm_alignr_epi8::<8>(t3, t3);
                let t5 = x86_64::_mm_add_epi16(t4, t3);
                let t6 = x86_64::_mm_packus_epi16(t5, t5);
                x86_64::_mm_cvtsi128_si32(t6).cast_unsigned()
            }
            _ => 0,
        }

        // let number_mask = dot_mask.not().bitand(mask);
        // let value = simd_sub(value, Simd::splat('0' as u8));

        // return hashcode;
    }
    // for _ in 0..new_line {
    //     let c = (value & 0xFF) as u8;
    //     value >>= 8;
    //
    //     if c == 0x0e {
    //         result |= (current_byte as u32) << shift_size;
    //         shift_size -= 8;
    //         current_byte = 0;
    //     } else {
    //         current_byte = current_byte.wrapping_mul(10).wrapping_add(c);
    //     }
    // }
    //
    // result | (current_byte as u32)
}

// #[inline(never)]
fn read_wide(bytes: &[u8]) -> u128 {
    if std::intrinsics::likely(bytes.len() >= 16) {
        u128::from_le_bytes(bytes[..16].try_into().unwrap())
    } else {
        let mut wide = 0u128;
        for &b in bytes.iter().take(16) {
            wide = (wide << 8) | (b as u128);
        }
        wide
    }
}

// #[inline(never)]
fn read_wide_be(bytes: &[u8]) -> u128 {
    if std::intrinsics::likely(bytes.len() >= 16) {
        u128::from_be_bytes(bytes[..16].try_into().unwrap())
    } else {
        let mut wide = 0u128;
        let mut offset = 128 - 8;
        for &b in bytes.iter().take(16) {
            wide |= (b as u128) << offset;
            offset -= 8;
        }
        wide
    }
}

// #[inline(never)]
fn newline_location(word: u128) -> u32 {
    let reversed = word ^ 0x0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A; // xor with \n\n\n...\n\n\n
    let code = (reversed.wrapping_sub(0x01010101010101010101010101010101))
        & (!reversed & 0x80808080808080808080808080808080);

    code.trailing_zeros() >> 3
}

// #[inline(never)]
fn worker(
    memory: &'static [u8],
    chunk_id: &'static AtomicUsize,
    bitset: &'static MaxBitSet,
    file_size: usize,
) {
    const CHUNK_SIZE: usize = 1 * 1024 * 1024;
    loop {
        let id = chunk_id.fetch_add(1, Ordering::SeqCst);
        if std::intrinsics::unlikely(id * CHUNK_SIZE >= memory.len()) {
            break;
        }
        let mut current_mem = &memory[(id * CHUNK_SIZE)..];
        let end = memory[std::cmp::min((id + 1) * CHUNK_SIZE, file_size)..].as_ptr();

        // Adjusting start
        if std::intrinsics::likely(id > 0) {
            let value = read_wide_be(current_mem);
            current_mem = &current_mem[((15 - newline_location(value)) + 1) as usize..];
        }

        parse_loop(bitset, current_mem, end);
    }
}

// #[inline(never)]
fn parse_loop(bitset: &MaxBitSet, mut current_mem: &[u8], end: *const u8) {
    while current_mem.as_ptr() < end {
        let value = read_wide_be(current_mem);
        let next_new = 15 - newline_location(value);
        bitset.set(num_simd(value, next_new));
        current_mem = &current_mem[((next_new + 1)as usize)..];
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

    let count = count_ips(&file);

    println!("Number of unique IP addresses: {}", count);
    Ok(())
}

fn count_ips(file: &File) -> u64 {
    let file_size = file.metadata().unwrap().len() as usize;

    // Create memory-mapped file
    let mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(file).unwrap() }));
    let map: &'static [u8] = unsafe { transmute::<&[u8], &'static [u8]>(mmap.as_ref()) };

    let chunk_id = Box::leak(Box::new(AtomicUsize::new(0)));
    let bitset = Box::leak(Box::new(MaxBitSet::new()));

    let mut handles = Vec::new();
    let num_threads = num_cpus::get();
    for _ in 0..num_threads {
        handles.push(thread::spawn(thread(map, chunk_id, bitset, file_size)));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    bitset.count()
}

fn thread(
    map: &'static [u8],
    chunk_id: &'static AtomicUsize,
    bitset: &'static MaxBitSet,
    file_size: usize,
) -> impl FnOnce() {
    move || {
        thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max).unwrap();
        worker(map, chunk_id, bitset, file_size);
    }
}

#[cfg(test)]
mod tests {
    use crate::{newline_location, num, num_simd, read_wide, read_wide_be};
    use rand::Rng;

    // #[test]
    // fn generate_test_file() {
    //     use rand::Rng;
    //     use std::fs::File;
    //     use std::io::{self, Write};
    //     use std::io::BufWriter;
    //
    //     /// Generates `n` random IP addresses and writes them to the specified file.
    //     fn generate_ips_and_write_to_file(filename: &str, n: usize) -> io::Result<()> {
    //         let mut file = BufWriter::new(File::create(filename)?);
    //         let mut rng = rand::thread_rng();
    //
    //         for _ in 0..n {
    //             // Generate a random IP address by creating four random octets.
    //             let ip = format!(
    //                 "{}.{}.{}.{}",
    //                 rng.gen::<u8>(),
    //                 rng.gen::<u8>(),
    //                 rng.gen::<u8>(),
    //                 rng.gen::<u8>()
    //             );
    //
    //             writeln!(file, "{}", ip)?;
    //         }
    //
    //         Ok(())
    //     }
    //
    //     let filename = "target/ips-100.txt";
    //     let n = 100;
    //
    //     generate_ips_and_write_to_file(filename, n).unwrap();
    //     println!("Generated {} IP addresses and wrote to {}", n, filename);
    // }

    /// Converts IP string to an integer bit index.
    fn ip_to_index(ip: &[u8]) -> Option<u32> {
        let parts: Vec<&[u8]> = ip.split(|&c| c == '.' as u8).collect();
        if parts.len() == 4 {
            let octets: Vec<u8> = parts
                .iter()
                .filter_map(|s| String::from_utf8_lossy(s).parse().ok())
                .collect();
            if octets.len() == 4 {
                Some(
                    ((octets[0] as u32) << 24)
                        | ((octets[1] as u32) << 16)
                        | ((octets[2] as u32) << 8)
                        | (octets[3] as u32),
                )
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Converts IP string to an integer bit index.
    fn ip_to_index_str(ip: &str) -> Option<u32> {
        let parts: Vec<&str> = ip.split('.').collect();
        if parts.len() == 4 {
            let octets: Vec<u8> = parts.iter().filter_map(|s| s.parse().ok()).collect();
            if octets.len() == 4 {
                Some(
                    ((octets[0] as u32) << 24)
                        | ((octets[1] as u32) << 16)
                        | ((octets[2] as u32) << 8)
                        | (octets[3] as u32),
                )
            } else {
                None
            }
        } else {
            None
        }
    }

    fn ip_to_index_str_rev(ip: &str) -> Option<u32> {
        let parts: Vec<&str> = ip.split('.').collect();
        if parts.len() == 4 {
            let octets: Vec<u8> = parts.iter().filter_map(|s| s.parse().ok()).collect();
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
        } else {
            None
        }
    }

    #[test]
    fn location_test_edges() {
        let location = newline_location(u128::from_le_bytes(*b"1.1.12.12123.34\n"));
        assert_eq!(location, 15);
        let location = newline_location(u128::from_le_bytes(*b"\n1.1.12.12123.34"));
        assert_eq!(location, 0);
    }

    #[test]
    fn location_test_1() {
        let location = newline_location(u128::from_le_bytes(*b"1.1.12.12\n123.34"));
        assert_eq!(location, 9);
    }

    #[test]
    fn location_test_2() {
        let location = newline_location(u128::from_le_bytes(*b"1.1.1.12\n123.34."));
        assert_eq!(location, 8);
    }

    #[test]
    fn location_test_3() {
        let location = newline_location(u128::from_le_bytes(*b"\n123.234.123.123"));
        assert_eq!(location, 0);
    }

    #[test]
    fn location_test_4() {
        let location = newline_location(u128::from_le_bytes(*b"1.13.123.123\n123"));
        assert_eq!(location, 12);
    }

    #[test]
    fn location_test_5() {
        let location = newline_location(u128::from_le_bytes(*b"123.123.123.123\n"));
        assert_eq!(location, 15);
    }

    #[test]
    fn location_test_6() {
        let location = newline_location(u128::from_le_bytes(*b"1.1.1.1\n1.1.1.1\n"));
        assert_eq!(location, 7);
    }

    #[test]
    fn location_test_7() {
        let location = newline_location(u128::from_le_bytes(*b"163.162.45.85\n1."));
        assert_eq!(location, 13);
    }

    #[test]
    fn location_test_8() {
        let location = newline_location(u128::from_le_bytes(*b"163.162.45.8521."));
        assert_eq!(location, 16);
    }

    #[test]
    fn parse_test_1() {
        let data = *b"163.162.45.85\n1.";
        let location = newline_location(u128::from_le_bytes(data));
        let number = num(read_wide(&data), location);
        let number_real = ip_to_index(&data[..(location as usize)]).unwrap();
        assert_eq!(number, number_real);
    }

    #[test]
    fn parse_test_2() {
        let data = *b"1.1.1.1\n2.2.2.2\n";
        let location = newline_location(u128::from_le_bytes(data));
        let number = num(read_wide(&data), location);
        assert_eq!(number, 0x01010101u32);
    }

    #[test]
    fn parse_test_3() {
        let data = *b"1.2.3.4\n2.2.2.2\n";
        let location = newline_location(u128::from_le_bytes(data));
        let number = num(read_wide(&data), location);
        assert_eq!(number, 0x01020304u32);
    }

    // #[test]
    // fn hash_test_1() {
    //     let mut rng = rand::thread_rng();
    //     let mut ip_str = move || {
    //         format!(
    //             "{}.{}.{}.{}",
    //             rng.gen::<u8>(),
    //             rng.gen::<u8>(),
    //             rng.gen::<u8>(),
    //             rng.gen::<u8>()
    //         )
    //     };
    //     let mut patter_to_hash = HashMap::new();
    //     for _ in 0..100000 {
    //         let ip1 = ip_str();
    //         let ip2 = ip_str();
    //         let data = format!("{ip1}\n{ip2}\n");
    //         let data_bytes = data.as_bytes()[0..16].try_into().unwrap();
    //         let location = newline_location(u128::from_le_bytes(data_bytes));
    //         let pattern = ip1.replace(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], "x");
    //         let hash = num2(read_wide_be(&data_bytes), location);
    //         if !patter_to_hash.contains_key(&pattern) {
    //             patter_to_hash.insert(pattern, hash);
    //         } else if patter_to_hash[&pattern] != hash {
    //             panic!(
    //                 "Hashes for patter {pattern} do not match {hash} {}\ntested ip memory segment {data}",
    //                 patter_to_hash[&pattern],
    //             );
    //         }
    //     }
    //     println!("{patter_to_hash:?}");
    // }

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
            let location = newline_location(u128::from_le_bytes(data_bytes));
            let parsed = num_simd(read_wide_be(&data_bytes), location);
            let parsed_correct = ip_to_index_str_rev(&ip1).unwrap();
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
            let location = newline_location(u128::from_le_bytes(data_bytes));
            let parsed = num_simd(read_wide_be(&data_bytes), location);
            let parsed_correct = ip_to_index_str_rev(&ip1).unwrap();
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
            let location = newline_location(u128::from_le_bytes(data_bytes));
            let parsed = num_simd(read_wide_be(&data_bytes), location);
            let parsed_correct = ip_to_index_str_rev(&ip1).unwrap();
            assert_eq!(format!("{parsed:#x}"), format!("{parsed_correct:#x}"));
        }
    }

    #[test]
    fn hash_test_2() {
        let data = String::from("22.70.70.13\n7.59.171.119");
        let data_bytes = data.as_bytes()[0..16].try_into().unwrap();
        let location = newline_location(u128::from_le_bytes(data_bytes));
        println!("{location}");
        let hash = num_simd(read_wide_be(&data_bytes), location);
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
            let data_bytes = data.as_bytes()[0..16].try_into().unwrap();
            let location = newline_location(u128::from_le_bytes(data_bytes));
            assert_eq!(
                ip1.len(),
                location as usize,
                "length mismatch {ip1:?} {location}"
            );
            let number = num(read_wide(&data_bytes), location);
            let number_real = ip_to_index_str(&ip1).unwrap();
            assert_eq!(number, number_real, "parsing this chunk failed '{data:?}'");
        }
    }
}
