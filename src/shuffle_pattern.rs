pub static PATTERNS_ID: [u8; 1229] = calculate().0;
pub static PATTERNS: [[u8; 16]; 82] = calculate().1;

const fn calculate_hash(lens: [u8; 4]) -> u32 {
    let mut dot_mask = 0u128;
    let mut offset = 0u32;
    let mut i = 0;
    while i < 4 {
        let len = lens[i];
        offset += len as u32 + 1;
        dot_mask |= 1u128.wrapping_shl(offset - 1);
        i += 1
    }
    ((dot_mask >> 5) ^ (dot_mask & 0x03ff)) as u32
}

const fn calculate_pattern(lens: [u8; 4]) -> [u8; 16] {
    let mut pattern = [255u8; 16];

    let mut offset = 0;
    let mut index = 0;
    while index < 4 {
        let len = lens[index];
        if len >= 3 {
            pattern[8 + 2 * index] = offset;
            offset += 1;
        }
        if len >= 2 {
            pattern[2 * index] = offset;
            offset += 1;
        }
        if len >= 1 {
            pattern[2 * index + 1] = offset;
            offset += 1;
        }
        offset += 1;
        index += 1;
    }
    pattern
}

const fn calculate() -> ([u8; 1229], [[u8; 16]; 82]) {
    let mut lookup1 = [81u8; 1229];
    let mut lookup2 = [[0u8; 16]; 82];
    let mut lookup2_i = 0;
    let mut i1: u8 = 1;
    while i1 < 4 {
        let mut i2: u8 = 1;
        while i2 < 4 {
            let mut i3: u8 = 1;
            while i3 < 4 {
                let mut i4: u8 = 1;
                while i4 < 4 {
                    let hash = calculate_hash([i1, i2, i3, i4]);
                    lookup1[hash as usize] = lookup2_i as u8;
                    let pattern = calculate_pattern([i1, i2, i3, i4]);
                    lookup2[lookup2_i] = pattern;
                    lookup2_i += 1;

                    i4 += 1;
                }
                i3 += 1;
            }
            i2 += 1;
        }
        i1 += 1;
    }
    lookup2[lookup2_i] = [0; 16];

    (lookup1, lookup2)
}
