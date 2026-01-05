const std = @import("std");
const expr = @import("../expr.zig");
const Heap = @import("../Heap.zig");
const utils = @import("../utils.zig");
const runtime_value = @import("../value.zig");
const builtins_bytestring = @import("bytestring.zig");

const ConstantType = expr.ConstantType;
const BigInt = expr.BigInt;
const Bytes = expr.Bytes;
const String = expr.String;

const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

const Secp256k1 = std.crypto.ecc.Secp256k1;
const secp256k1_half_order_be = blk: {
    var tmp: [32]u8 = undefined;
    const half_order = (Secp256k1.scalar.field_order - 1) / 2;
    std.mem.writeInt(u256, &tmp, half_order, .big);
    break :blk tmp;
};

inline fn isHighSecp256k1S(s: [32]u8) bool {
    return std.mem.order(u8, s[0..], secp256k1_half_order_be[0..]) == .gt;
}

const blst = @cImport({
    @cInclude("blst.h");
    @cInclude("blst_aux.h");
});

const empty_bytes = [_]u8{0};

const PackedBytes = struct {
    ptr: [*]const u8,
    len: usize,
};

fn materializeBytes(heap: *Heap, source: Bytes) PackedBytes {
    if (source.length == 0) {
        return .{
            .ptr = empty_bytes[0..].ptr,
            .len = 0,
        };
    }

    const byte_len: usize = @intCast(source.length);
    const word_len: u32 = @intCast((byte_len + 3) / 4);
    const buffer_words = heap.createArray(u32, word_len);
    const buffer_bytes: [*]u8 = @ptrCast(buffer_words);

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        buffer_bytes[i] = @truncate(source.bytes[i]);
    }

    // Zero any padding bytes (at most three) so future reads see deterministic data.
    var pad_index = byte_len;
    const padded_len = @as(usize, word_len) * @sizeOf(u32);
    while (pad_index < padded_len) : (pad_index += 1) {
        buffer_bytes[pad_index] = 0;
    }

    return .{
        .ptr = buffer_bytes,
        .len = byte_len,
    };
}

// Cryptography and hash functions
// Zig's std.crypto version of SHA2 currently produces incorrect digests when
// cross-compiled for our freestanding riscv32 target, so we ship a compact
// local implementation that matches the spec and accepts streamed byte input.
const Sha256Ctx = struct {
    h: [8]u32 = sha256_initial_state,
    buf: [64]u8 = undefined,
    buf_len: usize = 0,
    total_len: u64 = 0,

    fn init() Sha256Ctx {
        return Sha256Ctx{};
    }

    fn updateSlice(self: *Sha256Ctx, data: []const u8) void {
        var offset: usize = 0;
        self.total_len += data.len;

        if (self.buf_len != 0) {
            const space = 64 - self.buf_len;
            const to_copy = @min(space, data.len);
            @memcpy(self.buf[self.buf_len .. self.buf_len + to_copy], data[0..to_copy]);
            self.buf_len += to_copy;
            offset += to_copy;
            if (self.buf_len == 64) {
                self.processBlock(&self.buf);
                self.buf_len = 0;
            }
        }

        while (offset + 64 <= data.len) : (offset += 64) {
            const block_slice = data[offset..][0..64];
            const block_ptr: *const [64]u8 = @ptrCast(block_slice.ptr);
            self.processBlock(block_ptr);
        }

        const remaining = data.len - offset;
        if (remaining != 0) {
            @memcpy(self.buf[0..remaining], data[offset..]);
            self.buf_len = remaining;
        }
    }

    fn finalize(self: *Sha256Ctx, out: *[32]u8) void {
        // Append the single '1' bit (0x80 byte) then pad with zeros until we have room
        // for the 64-bit big-endian message length.
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if (self.buf_len > 56) {
            while (self.buf_len < 64) : (self.buf_len += 1) {
                self.buf[self.buf_len] = 0;
            }
            self.processBlock(&self.buf);
            self.buf_len = 0;
        }

        while (self.buf_len < 56) : (self.buf_len += 1) {
            self.buf[self.buf_len] = 0;
        }

        const bit_len: u64 = self.total_len * 8;
        var shift: usize = 0;
        while (shift < 8) : (shift += 1) {
            const idx = 7 - shift;
            const shift_amt: u6 = @intCast(idx * 8);
            self.buf[56 + shift] = @as(u8, @intCast((bit_len >> shift_amt) & 0xFF));
        }
        self.processBlock(&self.buf);
        self.buf_len = 0;

        for (self.h, 0..) |word, idx| {
            const base = idx * 4;
            out[base + 0] = @as(u8, @intCast(word >> 24));
            out[base + 1] = @as(u8, @intCast((word >> 16) & 0xFF));
            out[base + 2] = @as(u8, @intCast((word >> 8) & 0xFF));
            out[base + 3] = @as(u8, @intCast(word & 0xFF));
        }
    }

    inline fn rotr(value: u32, shift: u5) u32 {
        if (shift == 0) return value;
        const amt: u32 = shift;
        return (value >> amt) | (value << @as(u5, @intCast(32 - amt)));
    }

    fn processBlock(self: *Sha256Ctx, block: *const [64]u8) void {
        var w: [64]u32 = undefined;
        const bytes: [*]const u8 = @ptrCast(block);

        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const idx = i * 4;
            w[i] = (@as(u32, bytes[idx]) << 24) |
                (@as(u32, bytes[idx + 1]) << 16) |
                (@as(u32, bytes[idx + 2]) << 8) |
                @as(u32, bytes[idx + 3]);
        }

        while (i < 64) : (i += 1) {
            const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        var a = self.h[0];
        var b = self.h[1];
        var c = self.h[2];
        var d = self.h[3];
        var e = self.h[4];
        var f = self.h[5];
        var g = self.h[6];
        var h = self.h[7];

        i = 0;
        while (i < 64) : (i += 1) {
            const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const ch = (e & f) ^ (~e & g);
            const temp1 = h + s1 + ch + sha256_round_constants[i] + w[i];
            const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const maj = (a & b) ^ (a & c) ^ (b & c);
            const temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        self.h[0] += a;
        self.h[1] += b;
        self.h[2] += c;
        self.h[3] += d;
        self.h[4] += e;
        self.h[5] += f;
        self.h[6] += g;
        self.h[7] += h;
    }
};

const Sha512Ctx = struct {
    h: [8]u64 = sha512_initial_state,
    buf: [128]u8 = undefined,
    buf_len: usize = 0,
    total_len: u128 = 0,

    fn init() Sha512Ctx {
        return Sha512Ctx{};
    }

    fn updateSlice(self: *Sha512Ctx, data: []const u8) void {
        var offset: usize = 0;
        self.total_len += @as(u128, @intCast(data.len));

        if (self.buf_len != 0) {
            const space = 128 - self.buf_len;
            const to_copy = @min(space, data.len);
            @memcpy(self.buf[self.buf_len .. self.buf_len + to_copy], data[0..to_copy]);
            self.buf_len += to_copy;
            offset += to_copy;
            if (self.buf_len == 128) {
                self.processBlock(&self.buf);
                self.buf_len = 0;
            }
        }

        while (offset + 128 <= data.len) : (offset += 128) {
            const block_slice = data[offset..][0..128];
            const block_ptr: *const [128]u8 = @ptrCast(block_slice.ptr);
            self.processBlock(block_ptr);
        }

        const remaining = data.len - offset;
        if (remaining != 0) {
            @memcpy(self.buf[0..remaining], data[offset..]);
            self.buf_len = remaining;
        }
    }

    fn finalize(self: *Sha512Ctx, out: *[64]u8) void {
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if (self.buf_len > 112) {
            while (self.buf_len < 128) : (self.buf_len += 1) {
                self.buf[self.buf_len] = 0;
            }
            self.processBlock(&self.buf);
            self.buf_len = 0;
        }

        while (self.buf_len < 112) : (self.buf_len += 1) {
            self.buf[self.buf_len] = 0;
        }

        const bit_len: u128 = self.total_len * 8;
        var shift: usize = 0;
        while (shift < 16) : (shift += 1) {
            const idx = 15 - shift;
            const shift_amt: u7 = @intCast(idx * 8);
            self.buf[112 + shift] = @as(u8, @intCast((bit_len >> shift_amt) & 0xFF));
        }
        self.processBlock(&self.buf);
        self.buf_len = 0;

        for (self.h, 0..) |word, idx| {
            const base = idx * 8;
            out[base + 0] = @as(u8, @intCast(word >> 56));
            out[base + 1] = @as(u8, @intCast((word >> 48) & 0xFF));
            out[base + 2] = @as(u8, @intCast((word >> 40) & 0xFF));
            out[base + 3] = @as(u8, @intCast((word >> 32) & 0xFF));
            out[base + 4] = @as(u8, @intCast((word >> 24) & 0xFF));
            out[base + 5] = @as(u8, @intCast((word >> 16) & 0xFF));
            out[base + 6] = @as(u8, @intCast((word >> 8) & 0xFF));
            out[base + 7] = @as(u8, @intCast(word & 0xFF));
        }
    }

    inline fn rotr(value: u64, shift: u6) u64 {
        if (shift == 0) return value;
        const amt: u6 = shift;
        const inv: u6 = @intCast(@as(u7, 64) - @as(u7, amt));
        return (value >> amt) | (value << inv);
    }

    inline fn bigSigma0(x: u64) u64 {
        return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39);
    }

    inline fn bigSigma1(x: u64) u64 {
        return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41);
    }

    inline fn smallSigma0(x: u64) u64 {
        return rotr(x, 1) ^ rotr(x, 8) ^ (x >> 7);
    }

    inline fn smallSigma1(x: u64) u64 {
        return rotr(x, 19) ^ rotr(x, 61) ^ (x >> 6);
    }

    inline fn choose(x: u64, y: u64, z: u64) u64 {
        return (x & y) ^ ((~x) & z);
    }

    inline fn majority(x: u64, y: u64, z: u64) u64 {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    fn processBlock(self: *Sha512Ctx, block: *const [128]u8) void {
        var w: [80]u64 = undefined;
        const bytes = block[0..];

        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const base = i * 8;
            var word: u64 = 0;
            var j: usize = 0;
            while (j < 8) : (j += 1) {
                word = (word << 8) | @as(u64, bytes[base + j]);
            }
            w[i] = word;
        }

        i = 16;
        while (i < 80) : (i += 1) {
            const s0 = smallSigma0(w[i - 15]);
            const s1 = smallSigma1(w[i - 2]);
            w[i] = s1 +% w[i - 7] +% s0 +% w[i - 16];
        }

        var a = self.h[0];
        var b = self.h[1];
        var c = self.h[2];
        var d = self.h[3];
        var e = self.h[4];
        var f = self.h[5];
        var g = self.h[6];
        var h = self.h[7];

        i = 0;
        while (i < 80) : (i += 1) {
            const t1 = h +% bigSigma1(e) +% choose(e, f, g) +% sha512_round_constants[i] +% w[i];
            const t2 = bigSigma0(a) +% majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d +% t1;
            d = c;
            c = b;
            b = a;
            a = t1 +% t2;
        }

        self.h[0] = self.h[0] +% a;
        self.h[1] = self.h[1] +% b;
        self.h[2] = self.h[2] +% c;
        self.h[3] = self.h[3] +% d;
        self.h[4] = self.h[4] +% e;
        self.h[5] = self.h[5] +% f;
        self.h[6] = self.h[6] +% g;
        self.h[7] = self.h[7] +% h;
    }
};

const sha256_initial_state = [8]u32{
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
};

const sha256_round_constants = [64]u32{
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};

const sha512_initial_state = [8]u64{
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
};

const sha512_round_constants = [80]u64{
    0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
    0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
    0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
    0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
    0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
    0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
    0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
    0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
    0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
    0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
    0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30,
    0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
    0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
    0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
    0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
    0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
    0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
    0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
    0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
    0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817,
};

const sha3_rate_bytes: usize = 136;

const keccak_rho_offsets = [25]u8{
    0,  1,  62, 28, 27,
    36, 44, 6,  55, 20,
    3,  10, 43, 25, 39,
    41, 45, 15, 21, 8,
    18, 2,  61, 56, 14,
};

const keccak_round_constants = [24]u64{
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
};

inline fn rotl64(val: u64, shift: u6) u64 {
    if (shift == 0) return val; // avoid undefined >> 64 when rho offset is zero
    const inv: u6 = @intCast(@as(u7, 64) - @as(u7, shift));
    return (val << shift) | (val >> inv);
}

inline fn rotl32(val: u32, shift: u5) u32 {
    if (shift == 0) return val;
    const inv: u5 = @intCast(@as(u6, 32) - @as(u6, shift));
    return (val << shift) | (val >> inv);
}

fn keccakF1600(state: *[25]u64) void {
    var round: usize = 0;
    while (round < keccak_round_constants.len) : (round += 1) {
        var c: [5]u64 = undefined;
        var d: [5]u64 = undefined;

        var x: usize = 0;
        while (x < 5) : (x += 1) {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }

        x = 0;
        while (x < 5) : (x += 1) {
            d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
        }

        x = 0;
        while (x < 5) : (x += 1) {
            var y: usize = 0;
            while (y < 5) : (y += 1) {
                state[x + 5 * y] ^= d[x];
            }
        }

        var b: [25]u64 = undefined;
        var y: usize = 0;
        while (y < 5) : (y += 1) {
            x = 0;
            while (x < 5) : (x += 1) {
                const idx = x + 5 * y;
                const new_x = y;
                const new_y = (2 * x + 3 * y) % 5;
                b[new_x + 5 * new_y] = rotl64(state[idx], @intCast(keccak_rho_offsets[idx]));
            }
        }

        y = 0;
        while (y < 5) : (y += 1) {
            x = 0;
            while (x < 5) : (x += 1) {
                const idx = x + 5 * y;
                const b1 = b[((x + 1) % 5) + 5 * y];
                const b2 = b[((x + 2) % 5) + 5 * y];
                state[idx] = b[idx] ^ ((~b1) & b2);
            }
        }

        state[0] ^= keccak_round_constants[round];
    }
}

const Sha3_256Ctx = struct {
    state: [25]u64 = [_]u64{0} ** 25,
    pos: usize = 0,

    pub fn init() Sha3_256Ctx {
        return .{};
    }

    pub fn updateSlice(self: *Sha3_256Ctx, bytes: []const u8) void {
        for (bytes) |byte| {
            self.absorbByte(byte);
        }
    }

    fn absorbByte(self: *Sha3_256Ctx, byte: u8) void {
        const lane_index = self.pos / 8;
        const lane_shift: u6 = @intCast((self.pos % 8) * 8);
        self.state[lane_index] ^= @as(u64, byte) << lane_shift;
        self.pos += 1;

        if (self.pos == sha3_rate_bytes) {
            keccakF1600(&self.state);
            self.pos = 0;
        }
    }

    pub fn finalize(self: *Sha3_256Ctx, out: *[32]u8) void {
        self.finalizeWithSuffix(out, 0x06);
    }

    pub fn finalizeKeccak(self: *Sha3_256Ctx, out: *[32]u8) void {
        // Raw Keccak-256 (as used by Plutus) uses the 0x01 domain suffix.
        self.finalizeWithSuffix(out, 0x01);
    }

    fn finalizeWithSuffix(self: *Sha3_256Ctx, out: *[32]u8, suffix: u8) void {
        const lane_index = self.pos / 8;
        const lane_shift: u6 = @intCast((self.pos % 8) * 8);
        self.state[lane_index] ^= @as(u64, suffix) << lane_shift;

        const last_index = (sha3_rate_bytes - 1) / 8;
        const last_shift: u6 = @intCast(((sha3_rate_bytes - 1) % 8) * 8);
        self.state[last_index] ^= @as(u64, 0x80) << last_shift;

        keccakF1600(&self.state);
        self.pos = 0;
        self.squeeze(out[0..]);
    }

    fn squeeze(self: *Sha3_256Ctx, out: []u8) void {
        var produced: usize = 0;
        while (produced < out.len) : (produced += 1) {
            if (self.pos == sha3_rate_bytes) {
                keccakF1600(&self.state);
                self.pos = 0;
            }

            const lane_index = self.pos / 8;
            const lane_shift: u6 = @intCast((self.pos % 8) * 8);
            out[produced] = @truncate(self.state[lane_index] >> lane_shift);
            self.pos += 1;
        }
    }
};

// integer and basic bytestring builtins moved to runtime/src/builtins
pub fn sha2_256(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const input = args.value.unwrapBytestring();
    var hasher = Sha256Ctx.init();

    var chunk: [64]u8 = undefined;
    var chunk_len: usize = 0;
    const byte_len: usize = @intCast(input.length);

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        chunk[chunk_len] = @as(u8, @truncate(input.bytes[i]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }

    if (chunk_len != 0) {
        hasher.updateSlice(chunk[0..chunk_len]);
    }

    var digest: [32]u8 = undefined;
    hasher.finalize(&digest);

    var result = m.heap.createArray(u32, 32 + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 32;
    for (digest, 0..) |byte, idx| {
        result[4 + idx] = byte;
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn sha3_256(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const input = args.value.unwrapBytestring();
    var hasher = Sha3_256Ctx.init();

    const byte_len: usize = @intCast(input.length);
    var chunk: [64]u8 = undefined;
    var chunk_len: usize = 0;

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        chunk[chunk_len] = @as(u8, @truncate(input.bytes[i]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }

    if (chunk_len != 0) {
        hasher.updateSlice(chunk[0..chunk_len]);
    }

    var digest: [32]u8 = undefined;
    hasher.finalize(&digest);

    var result = m.heap.createArray(u32, 32 + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 32;
    for (digest, 0..) |byte, idx| {
        result[4 + idx] = byte;
    }

    return createConst(m.heap, @ptrCast(result));
}

const blake2b_iv = [_]u64{
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
};

const blake2b_sigma = [_][16]u8{
    [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    [_]u8{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    [_]u8{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    [_]u8{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    [_]u8{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    [_]u8{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    [_]u8{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    [_]u8{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    [_]u8{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    [_]u8{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    [_]u8{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
};

const Blake2bCtx = struct {
    h: [8]u64,
    t: [2]u64,
    buf: [128]u8,
    buf_len: usize,
    out_len: usize,

    fn init(out_len: usize) Blake2bCtx {
        var ctx = Blake2bCtx{
            .h = blake2b_iv,
            .t = .{ 0, 0 },
            .buf = [_]u8{0} ** 128,
            .buf_len = 0,
            .out_len = out_len,
        };
        ctx.h[0] ^= 0x01010000 ^ @as(u64, @intCast(out_len));
        return ctx;
    }

    fn update(self: *Blake2bCtx, data: []const u8) void {
        var offset: usize = 0;
        while (offset < data.len) {
            if (self.buf_len == self.buf.len) {
                self.addCounter(self.buf.len);
                self.compress(self.buf[0..], false);
                self.buf_len = 0;
            }

            const space = self.buf.len - self.buf_len;
            const take = @min(space, data.len - offset);
            @memcpy(
                self.buf[self.buf_len .. self.buf_len + take],
                data[offset .. offset + take],
            );
            self.buf_len += take;
            offset += take;
        }
    }

    fn final(self: *Blake2bCtx, out: []u8) void {
        @memset(self.buf[self.buf_len..], 0);
        self.addCounter(self.buf_len);
        self.compress(self.buf[0..], true);

        var i: usize = 0;
        while (i < out.len) : (i += 1) {
            const word = self.h[i / 8];
            const shift: u6 = @intCast((i % 8) * 8);
            out[i] = @as(u8, @truncate(word >> shift));
        }
    }

    fn addCounter(self: *Blake2bCtx, value: usize) void {
        const v: u64 = @intCast(value);
        self.t[0] +%= v;
        if (self.t[0] < v) {
            self.t[1] +%= 1;
        }
    }

    fn compress(self: *Blake2bCtx, block: []const u8, is_last: bool) void {
        var m: [16]u64 = undefined;
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const start = i * 8;
            m[i] = readLittle64(block, start);
        }

        var v: [16]u64 = undefined;
        i = 0;
        while (i < 8) : (i += 1) {
            v[i] = self.h[i];
            v[i + 8] = blake2b_iv[i];
        }

        v[12] ^= self.t[0];
        v[13] ^= self.t[1];
        if (is_last) v[14] = ~v[14];

        var round: usize = 0;
        while (round < 12) : (round += 1) {
            const s = blake2b_sigma[round];
            G(&v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            G(&v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            G(&v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            G(&v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            G(&v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            G(&v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            G(&v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            G(&v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }

        i = 0;
        while (i < 8) : (i += 1) {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }
};

fn G(v: *[16]u64, a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) void {
    v[a] = v[a] +% v[b] +% x;
    v[d] = std.math.rotr(u64, v[d] ^ v[a], 32);
    v[c] = v[c] +% v[d];
    v[b] = std.math.rotr(u64, v[b] ^ v[c], 24);
    v[a] = v[a] +% v[b] +% y;
    v[d] = std.math.rotr(u64, v[d] ^ v[a], 16);
    v[c] = v[c] +% v[d];
    v[b] = std.math.rotr(u64, v[b] ^ v[c], 63);
}

inline fn readLittle64(block: []const u8, start: usize) u64 {
    var accum: u64 = 0;
    var idx: usize = 0;
    while (idx < 8) : (idx += 1) {
        const shift: u6 = @intCast(idx * 8);
        accum |= @as(u64, block[start + idx]) << shift;
    }
    return accum;
}

fn blake2bHash(m: *BuiltinContext, args: *LinkedValues, comptime digest_bits: usize) *const Value {
    const input = args.value.unwrapBytestring();
    if (digest_bits % 8 != 0 or digest_bits == 0 or digest_bits > 512) {
        builtinEvaluationFailure();
    }

    const digest_len: usize = digest_bits / 8;
    var ctx = Blake2bCtx.init(digest_len);

    const byte_len: usize = @intCast(input.length);
    // ByteStrings store each byte in its own word, so repack into a dense
    // buffer before streaming into the hash state.
    var chunk: [128]u8 = undefined;
    var chunk_len: usize = 0;

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        chunk[chunk_len] = @as(u8, @truncate(input.bytes[i]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            ctx.update(chunk[0..]);
            chunk_len = 0;
        }
    }

    if (chunk_len != 0) {
        ctx.update(chunk[0..chunk_len]);
    }

    var digest: [64]u8 = undefined;
    ctx.final(digest[0..digest_len]);

    const digest_words: u32 = @intCast(digest_len);
    const allocation = builtins_bytestring.initByteStringAllocation(m.heap, digest_words);
    for (digest[0..digest_len], 0..) |byte, idx| {
        allocation.data_words[idx] = byte;
    }

    return createConst(m.heap, @ptrCast(allocation.constant_words));
}

pub fn blake2b_256(m: *BuiltinContext, args: *LinkedValues) *const Value {
    return blake2bHash(m, args, 256);
}

fn makeBoolConst(m: *BuiltinContext, value: bool) *const Value {
    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(value);
    return createConst(m.heap, @ptrCast(result));
}

pub fn verifyEd25519Signature(m: *BuiltinContext, args: *LinkedValues) *const Value {
    // Arguments come in reverse order: signature, message, public_key
    const signature_bs = args.value.unwrapBytestring();
    const message_bs = args.next.?.value.unwrapBytestring();
    const public_key_bs = args.next.?.next.?.value.unwrapBytestring();

    // Ed25519 requires exactly 32 bytes for public key and 64 bytes for signature
    if (public_key_bs.length != 32) {
        builtinEvaluationFailure();
    }
    if (signature_bs.length != 64) {
        builtinEvaluationFailure();
    }
    // Extract public key bytes
    var public_key_bytes: [32]u8 = undefined;
    for (0..32) |i| {
        public_key_bytes[i] = @as(u8, @truncate(public_key_bs.bytes[i]));
    }

    // Extract signature bytes
    var signature_bytes: [64]u8 = undefined;
    for (0..64) |i| {
        signature_bytes[i] = @as(u8, @truncate(signature_bs.bytes[i]));
    }

    const message_len: usize = @intCast(message_bs.length);

    const Ed25519 = std.crypto.sign.Ed25519;
    const Curve = Ed25519.Curve;

    // Use the local SHA-512 to stay correct on riscv32 (std.crypto's version miscompiles).
    var hasher = Sha512Ctx.init();
    hasher.updateSlice(signature_bytes[0..32]);
    hasher.updateSlice(public_key_bytes[0..]);

    var chunk: [256]u8 = undefined;
    var chunk_len: usize = 0;
    var idx: usize = 0;
    while (idx < message_len) : (idx += 1) {
        chunk[chunk_len] = @as(u8, @truncate(message_bs.bytes[idx]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }
    if (chunk_len != 0) {
        hasher.updateSlice(chunk[0..chunk_len]);
    }

    var hram64: [64]u8 = undefined;
    hasher.finalize(&hram64);
    const h_scalar = Curve.scalar.reduce64(hram64);

    var s_scalar: [32]u8 = undefined;
    @memcpy(&s_scalar, signature_bytes[32..]);
    Curve.scalar.rejectNonCanonical(s_scalar) catch {
        return makeBoolConst(m, false);
    };

    var r_bytes: [32]u8 = undefined;
    @memcpy(&r_bytes, signature_bytes[0..32]);
    Curve.rejectNonCanonical(r_bytes) catch {
        return makeBoolConst(m, false);
    };

    const pk_point = Curve.fromBytes(public_key_bytes) catch {
        return makeBoolConst(m, false);
    };
    pk_point.rejectIdentity() catch {
        return makeBoolConst(m, false);
    };

    const expected_r = Curve.fromBytes(r_bytes) catch {
        return makeBoolConst(m, false);
    };
    expected_r.rejectIdentity() catch {
        return makeBoolConst(m, false);
    };

    const sb_minus_kA = Curve.mulDoubleBasePublic(
        Curve.basePoint,
        s_scalar,
        pk_point.neg(),
        h_scalar,
    ) catch {
        return makeBoolConst(m, false);
    };

    const diff = expected_r.sub(sb_minus_kA);
    // ZIP-215 compatibility: accept signatures when the difference lies in the
    // small torsion subgroup (rejectLowOrder raises for non-torsion points).
    diff.rejectLowOrder() catch {
        return makeBoolConst(m, true);
    };

    return makeBoolConst(m, false);
}

const StringBytes = struct {
    byte_len: u32,
    is_packed: bool,
};

fn analyzeString(str: String) StringBytes {
    var is_packed = false;
    var probe: u32 = 0;
    while (probe < str.length) : (probe += 1) {
        if (str.bytes[probe] > 0xFF) {
            is_packed = true;
            break;
        }
    }

    return .{
        .byte_len = if (is_packed) packedLength(str) else unpackedLength(str),
        .is_packed = is_packed,
    };
}

fn unpackedLength(str: String) u32 {
    var len = str.length;
    while (len > 0 and (str.bytes[len - 1] & 0xFF) == 0) {
        len -= 1;
    }
    return len;
}

fn packedLength(str: String) u32 {
    var total: u32 = str.length * 4;
    if (total == 0) return 0;

    var idx = str.length;
    while (idx > 0) : (idx -= 1) {
        const word = str.bytes[idx - 1];
        var shift: u5 = 24;
        while (true) {
            const byte_val = (word >> shift) & 0xFF;
            if (byte_val == 0) {
                total -= 1;
                if (total == 0) return 0;
            } else {
                return total;
            }

            if (shift == 0) break;
            shift -= 8;
        }
    }

    return 0;
}

const ByteWriter = struct {
    dst: [*]u32, // dest words, one byte per word (unpacked form)
    index: u32 = 0,

    fn init(dst: [*]u32) ByteWriter {
        return .{ .dst = dst };
    }

    fn write(self: *ByteWriter, value: u8) void {
        self.dst[self.index] = value;
        self.index += 1;
    }
};

fn appendStringBytes(writer: *ByteWriter, str: String, view: StringBytes) void {
    if (view.byte_len == 0) return;

    if (!view.is_packed) {
        var i: u32 = 0;
        while (i < view.byte_len) : (i += 1) {
            writer.write(@truncate(str.bytes[i]));
        }
        return;
    }

    var remaining = view.byte_len;
    var word_idx: u32 = 0;
    var byte_idx: u2 = 0;

    while (remaining > 0 and word_idx < str.length) {
        const word = str.bytes[word_idx];
        const shift: u5 = @intCast(@as(u32, byte_idx) * 8);
        const byte_val = @as(u8, @truncate((word >> shift) & 0xFF));
        writer.write(byte_val);

        remaining -= 1;
        byte_idx += 1;
        if (byte_idx == 4) {
            byte_idx = 0;
            word_idx += 1;
        }
    }
}

fn extractStringByte(str: String, view: StringBytes, byte_index: u32) u8 {
    if (byte_index >= view.byte_len) return 0;

    if (!view.is_packed) {
        // Unpacked: one byte per word
        return @truncate(str.bytes[byte_index]);
    } else {
        // Packed: 4 bytes per word in little-endian
        const word_idx = byte_index / 4;
        const byte_in_word = byte_index % 4;
        const word = str.bytes[word_idx];
        const shift: u5 = @intCast(byte_in_word * 8);
        return @as(u8, @truncate((word >> shift) & 0xFF));
    }
}


pub fn verifyEcdsaSecp256k1Signature(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const signature_bs = args.value.unwrapBytestring();
    const message_bs = args.next.?.value.unwrapBytestring();
    const public_key_bs = args.next.?.next.?.value.unwrapBytestring();

    var pk_bytes: [33]u8 = undefined;
    if (!unpackWordPackedBytes(33, public_key_bs, &pk_bytes)) {
        builtinEvaluationFailure();
    }
    var message_bytes: [32]u8 = undefined;
    if (!unpackWordPackedBytes(32, message_bs, &message_bytes)) {
        builtinEvaluationFailure();
    }
    var signature_bytes: [64]u8 = undefined;
    if (!unpackWordPackedBytes(64, signature_bs, &signature_bytes)) {
        builtinEvaluationFailure();
    }

    var pub_point = Secp256k1.fromSec1(pk_bytes[0..]) catch {
        builtinEvaluationFailure();
    };
    pub_point.rejectIdentity() catch {
        builtinEvaluationFailure();
    };

    var r_bytes: [32]u8 = undefined;
    var s_bytes: [32]u8 = undefined;
    @memcpy(r_bytes[0..], signature_bytes[0..32]);
    @memcpy(s_bytes[0..], signature_bytes[32..64]);

    Secp256k1.scalar.rejectNonCanonical(r_bytes, .big) catch {
        builtinEvaluationFailure();
    };
    Secp256k1.scalar.rejectNonCanonical(s_bytes, .big) catch {
        builtinEvaluationFailure();
    };

    const Scalar = Secp256k1.scalar.Scalar;
    const r_scalar = Scalar.fromBytes(r_bytes, .big) catch unreachable;
    const s_scalar = Scalar.fromBytes(s_bytes, .big) catch unreachable;
    if (r_scalar.isZero() or s_scalar.isZero()) {
        return makeBoolConst(m, false);
    }

    // High-S signatures are rejected (BIP-146 / Plutus requirement).
    if (isHighSecp256k1S(s_bytes)) {
        return makeBoolConst(m, false);
    }

    var digest_pad: [64]u8 = undefined;
    @memset(digest_pad[0..], 0);
    @memcpy(digest_pad[32..], message_bytes[0..]);
    const z_bytes = Secp256k1.scalar.reduce64(digest_pad, .big);
    const z_scalar = Scalar.fromBytes(z_bytes, .big) catch unreachable;

    const w_scalar = s_scalar.invert();
    const u1_bytes = z_scalar.mul(w_scalar).toBytes(.big);
    const u2_bytes = r_scalar.mul(w_scalar).toBytes(.big);

    const r_point = Secp256k1.mulDoubleBasePublic(
        Secp256k1.basePoint,
        u1_bytes,
        pub_point,
        u2_bytes,
        .big,
    ) catch {
        return makeBoolConst(m, false);
    };

    const r_xy = r_point.affineCoordinates();
    const rx_bytes = r_xy.x.toBytes(.big);
    var rx_pad: [64]u8 = undefined;
    @memset(rx_pad[0..], 0);
    @memcpy(rx_pad[32..], rx_bytes[0..]);
    const rx_mod = Secp256k1.scalar.reduce64(rx_pad, .big);

    const is_valid = std.mem.eql(u8, rx_mod[0..], r_bytes[0..]);
    return makeBoolConst(m, is_valid);
}

pub fn verifySchnorrSecp256k1Signature(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const signature_bs = args.value.unwrapBytestring();
    const message_bs = args.next.?.value.unwrapBytestring();
    const public_key_bs = args.next.?.next.?.value.unwrapBytestring();

    if (public_key_bs.length != 32 or signature_bs.length != 64) {
        builtinEvaluationFailure();
    }

    var pk_bytes: [32]u8 = undefined;
    var idx: usize = 0;
    while (idx < 32) : (idx += 1) {
        pk_bytes[idx] = @as(u8, @truncate(public_key_bs.bytes[idx]));
    }

    var r_bytes: [32]u8 = undefined;
    var s_bytes: [32]u8 = undefined;
    idx = 0;
    while (idx < 32) : (idx += 1) {
        r_bytes[idx] = @as(u8, @truncate(signature_bs.bytes[idx]));
        s_bytes[idx] = @as(u8, @truncate(signature_bs.bytes[32 + idx]));
    }

    Secp256k1.scalar.rejectNonCanonical(s_bytes, .big) catch {
        return makeBoolConst(m, false);
    };
    _ = Secp256k1.Fe.fromBytes(r_bytes, .big) catch {
        return makeBoolConst(m, false);
    };

    // Plutus requires invalid public keys to raise an evaluation failure rather than return False.
    const pk_x = Secp256k1.Fe.fromBytes(pk_bytes, .big) catch {
        builtinEvaluationFailure();
    };
    const pk_y = Secp256k1.recoverY(pk_x, false) catch {
        builtinEvaluationFailure();
    };
    var pub_point = Secp256k1.fromAffineCoordinates(.{ .x = pk_x, .y = pk_y }) catch {
        builtinEvaluationFailure();
    };
    pub_point.rejectIdentity() catch {
        builtinEvaluationFailure();
    };

    // BIP-340 uses a tagged SHA-256 challenge over r || pk || message.
    const challenge_tag = "BIP0340/challenge";
    var tag_hash: [32]u8 = undefined;
    var tag_hasher = Sha256Ctx.init();
    tag_hasher.updateSlice(challenge_tag);
    tag_hasher.finalize(&tag_hash);

    var challenge_hasher = Sha256Ctx.init();
    challenge_hasher.updateSlice(tag_hash[0..]);
    challenge_hasher.updateSlice(tag_hash[0..]);
    challenge_hasher.updateSlice(r_bytes[0..]);
    challenge_hasher.updateSlice(pk_bytes[0..]);

    const message_len: usize = @intCast(message_bs.length);
    var chunk: [256]u8 = undefined;
    var chunk_len: usize = 0;
    var msg_idx: usize = 0;
    while (msg_idx < message_len) : (msg_idx += 1) {
        chunk[chunk_len] = @as(u8, @truncate(message_bs.bytes[msg_idx]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            challenge_hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }
    if (chunk_len != 0) {
        challenge_hasher.updateSlice(chunk[0..chunk_len]);
    }

    var challenge: [32]u8 = undefined;
    challenge_hasher.finalize(&challenge);

    var challenge_input: [64]u8 = undefined;
    @memset(challenge_input[0..], 0);
    @memcpy(challenge_input[32..], challenge[0..]);
    const e_bytes = Secp256k1.scalar.reduce64(challenge_input, .big);
    const neg_e = Secp256k1.scalar.neg(e_bytes, .big) catch {
        return makeBoolConst(m, false);
    };

    // Compute R = s*G - e*P per BIP-340; mulDoubleBase sums scalar multiples.
    const r_point = Secp256k1.mulDoubleBasePublic(
        Secp256k1.basePoint,
        s_bytes,
        pub_point,
        neg_e,
        .big,
    ) catch {
        return makeBoolConst(m, false);
    };

    const r_xy = r_point.affineCoordinates();
    if (r_xy.y.isOdd()) {
        return makeBoolConst(m, false);
    }
    const rx_bytes = r_xy.x.toBytes(.big);
    if (!std.mem.eql(u8, rx_bytes[0..], r_bytes[0..])) {
        return makeBoolConst(m, false);
    }

    return makeBoolConst(m, true);
}

// BLS Builtins

fn integerToScalarBytes(bi: BigInt) [32]u8 {
    const scalar = integerToBlstScalar(bi);
    return scalar.b;
}

fn integerToBlstScalar(bi: BigInt) blst.blst_scalar {
    // Reduce the (possibly huge or negative) integer modulo the scalar field
    // using BLST's wide-input helpers so we match Plutus semantics.
    var scalar = std.mem.zeroes(blst.blst_scalar);

    const word_len: usize = @intCast(bi.length);
    if (word_len != 0) {
        const words = bi.words[0..word_len];
        const magnitude = std.mem.sliceAsBytes(words);
        _ = blst.blst_scalar_from_le_bytes(&scalar, magnitude.ptr, magnitude.len);
    }

    if (bi.sign != 0) {
        var tmp_fr: blst.blst_fr = undefined;
        blst.blst_fr_from_scalar(&tmp_fr, &scalar);
        blst.blst_fr_cneg(&tmp_fr, &tmp_fr, true);
        blst.blst_scalar_from_fr(&scalar, &tmp_fr);
    }

    return scalar;
}

const bls12_381_fp_modulus_be = [_]u8{
    0x1a, 0x01, 0x11, 0xea, 0x39, 0x7f, 0xe6, 0x9a,
    0x4b, 0x1b, 0xa7, 0xb6, 0x43, 0x4b, 0xac, 0xd7,
    0x64, 0x77, 0x4b, 0x84, 0xf3, 0x85, 0x12, 0xbf,
    0x67, 0x30, 0xd2, 0xa0, 0xf6, 0xb0, 0xf6, 0x24,
    0x1e, 0xab, 0xff, 0xfe, 0xb1, 0x53, 0xff, 0xff,
    0xb9, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xaa, 0xab,
};

fn decodeFpFromBendian(dst: *blst.blst_fp, chunk: [*]const u8) bool {
    blst.blst_fp_from_bendian(dst, chunk);
    // Enforce canonical encoding because blst_fp_from_bendian never fails.
    return std.mem.lessThan(u8, chunk[0..48], bls12_381_fp_modulus_be[0..]);
}

fn blst_fp12_from_bendian(ret: *blst.blst_fp12, bytes: [*]const u8) c_int {
    var offset: usize = 0;
    // Inverse of blst_bendian_from_fp12: iterate fp2 limbs first and interleave fp6 components
    // so we consume the 12 field elements in the same order they were emitted.
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            if (!decodeFpFromBendian(&ret.fp6[j].fp2[i].fp[0], bytes + offset))
                return blst.BLST_BAD_ENCODING;
            offset += 48;
            if (!decodeFpFromBendian(&ret.fp6[j].fp2[i].fp[1], bytes + offset))
                return blst.BLST_BAD_ENCODING;
            offset += 48;
        }
    }

    // Miller loop outputs are not guaranteed to lie in the final exponent subgroup,
    // so we only validate per-coordinate decoding here.
    return blst.BLST_SUCCESS;
}

pub fn bls12_381_G1_Add(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG1();
    const p = args.next.?.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);
    var q_bytes: [48]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);
    var aff_q: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_q: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_q, &aff_q);

    var point_r: blst.blst_p1 = undefined;
    blst.blst_p1_add(&point_r, &point_p, &point_q);

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_Neg(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    blst.blst_p1_cneg(&point_p, true);

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_p);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_ScalarMul(m: *BuiltinContext, args: *LinkedValues) *const Value {
    // Spec orders arguments as (G1, scalar); the newest (scalar) is in the tail.
    const p = args.value.unwrapG1();
    const scalar = args.next.?.value.unwrapInteger();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    const scalar_words = integerToBlstScalar(scalar);

    var point_r: blst.blst_p1 = undefined;
    blst.blst_p1_mult(
        &point_r,
        &point_p,
        &scalar_words.b,
        @sizeOf(blst.blst_scalar) * 8,
    );

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_Equal(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG1();
    const p = args.next.?.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);
    var q_bytes: [48]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_q: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_q, &aff_q);

    const equal = blst.blst_p1_is_equal(&point_p, &point_q);

    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(equal);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_Compress(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var out_bytes: [48]u8 = undefined;
    blst.blst_p1_compress(&out_bytes, &point_p);

    var result = m.heap.createArray(u32, 52);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 48;
    for (0..48) |i| {
        result[4 + i] = out_bytes[i];
    }
    return createConst(m.heap, @ptrCast(result));
}

// ByteStrings are stored with one byte value per u32 word (unpacked format).
// This function extracts them into a continuous byte array for BLST library calls.
fn unpackWordPackedBytes(comptime expected_len: usize, bs: Bytes, out: *[expected_len]u8) bool {
    const byte_count: usize = @intCast(bs.length);
    if (byte_count != expected_len) {
        return false;
    }

    var i: usize = 0;
    while (i < byte_count) : (i += 1) {
        out[i] = @truncate(bs.bytes[i]);
    }
    return true;
}

pub fn bls12_381_G1_Uncompress(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bs = args.value.unwrapBytestring();
    var in_bytes: [48]u8 = undefined;
    if (!unpackWordPackedBytes(48, bs, &in_bytes)) {
        builtinEvaluationFailure();
    }

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &in_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);
    if (!blst.blst_p1_in_g1(&point_p)) {
        builtinEvaluationFailure();
    }

    var canonical: [48]u8 = undefined;
    blst.blst_p1_compress(&canonical, &point_p);
    if (!std.mem.eql(u8, canonical[0..], in_bytes[0..])) {
        // Input encoding must round-trip through compression.
        builtinEvaluationFailure();
    }

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    @memcpy(out_bytes[0..48], canonical[0..]);
    return createConst(m.heap, @ptrCast(result));
}
// Alias kept so callers using the spec's "decompress" name still resolve.
pub const bls12_381_G1_Decompress = bls12_381_G1_Uncompress;

pub fn bls12_381_G1_HashToGroup(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const dst = args.value.unwrapBytestring();
    const msg = args.next.?.value.unwrapBytestring();

    const dst_packed = materializeBytes(m.heap, dst);
    const msg_packed = materializeBytes(m.heap, msg);
    if (dst_packed.len > 255) {
        // PLT spec enforces DST <= 255 bytes.
        builtinEvaluationFailure();
    }

    var point_r: blst.blst_p1 = undefined;
    blst.blst_hash_to_g1(&point_r, msg_packed.ptr, msg_packed.len, dst_packed.ptr, dst_packed.len, null, 0);

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Add(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG2();
    const p = args.next.?.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);
    var q_bytes: [96]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_q: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_q, &aff_q);

    var point_r: blst.blst_p2 = undefined;
    blst.blst_p2_add(&point_r, &point_p, &point_q);

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Neg(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    blst.blst_p2_cneg(&point_p, true);

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_p);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_ScalarMul(m: *BuiltinContext, args: *LinkedValues) *const Value {
    // Spec orders arguments as (G2, scalar); the newest (scalar) is in the tail.
    const p = args.value.unwrapG2();
    const scalar = args.next.?.value.unwrapInteger();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    const scalar_words = integerToBlstScalar(scalar);

    var point_r: blst.blst_p2 = undefined;
    blst.blst_p2_mult(
        &point_r,
        &point_p,
        &scalar_words.b,
        @sizeOf(blst.blst_scalar) * 8,
    );

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Equal(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG2();
    const p = args.next.?.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);
    var q_bytes: [96]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_q: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_q, &aff_q);

    const equal = blst.blst_p2_is_equal(&point_p, &point_q);

    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(equal);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Compress(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var out_bytes: [96]u8 = undefined;
    blst.blst_p2_compress(&out_bytes, &point_p);

    var result = m.heap.createArray(u32, 100);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 96;
    for (0..96) |i| {
        result[4 + i] = out_bytes[i];
    }
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Uncompress(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bs = args.value.unwrapBytestring();
    var in_bytes: [96]u8 = undefined;
    if (!unpackWordPackedBytes(96, bs, &in_bytes)) {
        builtinEvaluationFailure();
    }

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &in_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);
    if (!blst.blst_p2_in_g2(&point_p)) {
        builtinEvaluationFailure();
    }

    var canonical: [96]u8 = undefined;
    blst.blst_p2_compress(&canonical, &point_p);
    if (!std.mem.eql(u8, canonical[0..], in_bytes[0..])) {
        builtinEvaluationFailure();
    }

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    @memcpy(out_bytes[0..96], canonical[0..]);
    return createConst(m.heap, @ptrCast(result));
}
// Alias kept so callers using the spec's "decompress" name still resolve.
pub const bls12_381_G2_Decompress = bls12_381_G2_Uncompress;

pub fn bls12_381_G2_HashToGroup(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const dst = args.value.unwrapBytestring();
    const msg = args.next.?.value.unwrapBytestring();

    const dst_packed = materializeBytes(m.heap, dst);
    const msg_packed = materializeBytes(m.heap, msg);
    if (dst_packed.len > 255) {
        builtinEvaluationFailure();
    }

    var point_r: blst.blst_p2 = undefined;
    blst.blst_hash_to_g2(&point_r, msg_packed.ptr, msg_packed.len, dst_packed.ptr, dst_packed.len, null, 0);

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_MillerLoop(m: *BuiltinContext, args: *LinkedValues) *const Value {
    // Spec orders arguments as (G1, G2); the newest (G2) sits at the list head.
    const g2 = args.value.unwrapG2();
    const g1 = args.next.?.value.unwrapG1();

    var g1_bytes: [48]u8 = undefined;
    @memcpy(&g1_bytes, g1.bytes[0..48]);
    var g2_bytes: [96]u8 = undefined;
    @memcpy(&g2_bytes, g2.bytes[0..96]);

    var aff_g1: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_g1, &g1_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var chk_g1: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&chk_g1, &aff_g1);
    if (!blst.blst_p1_in_g1(&chk_g1)) {
        builtinEvaluationFailure();
    }

    var aff_g2: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_g2, &g2_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var chk_g2: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&chk_g2, &aff_g2);
    if (!blst.blst_p2_in_g2(&chk_g2)) {
        builtinEvaluationFailure();
    }

    var ml: blst.blst_fp12 = undefined;
    blst.blst_miller_loop(&ml, &aff_g2, &aff_g1);

    var result = m.heap.createArray(u32, 147);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.mlResultType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_bendian_from_fp12(out_bytes, &ml);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_MulMlResult(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const b = args.value.unwrapMlResult();
    const a = args.next.?.value.unwrapMlResult();

    var a_bytes: [576]u8 = undefined;
    @memcpy(&a_bytes, a.bytes[0..576]);
    var b_bytes: [576]u8 = undefined;
    @memcpy(&b_bytes, b.bytes[0..576]);

    var fp_a: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_a, &a_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }

    var fp_b: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_b, &b_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }

    var fp_r: blst.blst_fp12 = undefined;
    blst.blst_fp12_mul(&fp_r, &fp_a, &fp_b);

    var result = m.heap.createArray(u32, 147);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.mlResultType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_bendian_from_fp12(out_bytes, &fp_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_FinalVerify(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const ml2 = args.value.unwrapMlResult();
    const ml1 = args.next.?.value.unwrapMlResult();

    var ml1_bytes: [576]u8 = undefined;
    @memcpy(&ml1_bytes, ml1.bytes[0..576]);
    var ml2_bytes: [576]u8 = undefined;
    @memcpy(&ml2_bytes, ml2.bytes[0..576]);

    var fp_ml1: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_ml1, &ml1_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }

    var fp_ml2: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_ml2, &ml2_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }

    const res = blst.blst_fp12_finalverify(&fp_ml1, &fp_ml2);

    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(res);
    return createConst(m.heap, @ptrCast(result));
}

pub fn keccak_256(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const input = args.value.unwrapBytestring();
    var hasher = Sha3_256Ctx.init();

    const byte_len: usize = @intCast(input.length);
    var chunk: [64]u8 = undefined;
    var chunk_len: usize = 0;

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        chunk[chunk_len] = @as(u8, @truncate(input.bytes[i]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }

    if (chunk_len != 0) {
        hasher.updateSlice(chunk[0..chunk_len]);
    }

    var digest: [32]u8 = undefined;
    hasher.finalizeKeccak(&digest);

    var result = m.heap.createArray(u32, 32 + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 32;
    for (digest, 0..) |byte, idx| {
        result[4 + idx] = byte;
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn blake2b_224(m: *BuiltinContext, args: *LinkedValues) *const Value {
    return blake2bHash(m, args, 224);
}

// Conversions
// extended bytestring builtins moved to runtime/src/builtins
// RIPEMD-160 Context and Implementation
const RipeMd160Ctx = struct {
    h: [5]u32 = ripemd160_initial_state,
    buf: [64]u8 = [_]u8{0} ** 64,
    buf_len: usize = 0,
    total_len: u64 = 0,

    fn init() RipeMd160Ctx {
        return RipeMd160Ctx{};
    }

    fn updateSlice(self: *RipeMd160Ctx, data: []const u8) void {
        var offset: usize = 0;
        self.total_len += data.len;

        if (self.buf_len != 0) {
            const space = 64 - self.buf_len;
            const to_copy = @min(space, data.len);
            @memcpy(self.buf[self.buf_len .. self.buf_len + to_copy], data[0..to_copy]);
            self.buf_len += to_copy;
            offset += to_copy;
            if (self.buf_len == 64) {
                self.processBlock(&self.buf);
                self.buf_len = 0;
            }
        }

        while (offset + 64 <= data.len) : (offset += 64) {
            const block_slice = data[offset..][0..64];
            const block_ptr: *const [64]u8 = @ptrCast(block_slice.ptr);
            self.processBlock(block_ptr);
        }

        const remaining = data.len - offset;
        if (remaining != 0) {
            @memcpy(self.buf[0..remaining], data[offset..]);
            self.buf_len = remaining;
        }
    }

    fn finalize(self: *RipeMd160Ctx, out: *[20]u8) void {
        // Append the single '1' bit (0x80 byte)
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        // If not enough room for the length, pad and process, then start a new block
        if (self.buf_len > 56) {
            while (self.buf_len < 64) : (self.buf_len += 1) {
                self.buf[self.buf_len] = 0;
            }
            self.processBlock(&self.buf);
            self.buf_len = 0;
        }

        // Pad with zeros until we have 56 bytes
        while (self.buf_len < 56) : (self.buf_len += 1) {
            self.buf[self.buf_len] = 0;
        }

        // Append the message length in bits as a 64-bit little-endian integer
        const bit_len: u64 = self.total_len * 8;
        var shift: usize = 0;
        while (shift < 8) : (shift += 1) {
            const shift_amt: u6 = @intCast(shift * 8);
            self.buf[56 + shift] = @as(u8, @intCast((bit_len >> shift_amt) & 0xFF));
        }
        self.processBlock(&self.buf);

        // Output hash in little-endian format
        for (self.h, 0..) |word, idx| {
            const base = idx * 4;
            out[base + 0] = @as(u8, @intCast(word & 0xFF));
            out[base + 1] = @as(u8, @intCast((word >> 8) & 0xFF));
            out[base + 2] = @as(u8, @intCast((word >> 16) & 0xFF));
            out[base + 3] = @as(u8, @intCast(word >> 24));
        }
    }

    inline fn rotl(value: u32, shift: u5) u32 {
        if (shift == 0) return value;
        const amt: u32 = shift;
        return (value << shift) | (value >> @as(u5, @intCast(32 - amt)));
    }

    fn processBlock(self: *RipeMd160Ctx, block: *const [64]u8) void {
        // Parse block into 16 little-endian 32-bit words
        var x: [16]u32 = undefined;
        const bytes: [*]const u8 = @ptrCast(block);

        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const idx = i * 4;
            x[i] = @as(u32, bytes[idx]) |
                (@as(u32, bytes[idx + 1]) << 8) |
                (@as(u32, bytes[idx + 2]) << 16) |
                (@as(u32, bytes[idx + 3]) << 24);
        }

        // Initialize working variables
        var al = self.h[0];
        var bl = self.h[1];
        var cl = self.h[2];
        var dl = self.h[3];
        var el = self.h[4];

        var ar = self.h[0];
        var br = self.h[1];
        var cr = self.h[2];
        var dr = self.h[3];
        var er = self.h[4];

        // Left rounds
        i = 0;
        while (i < 80) : (i += 1) {
            const f = ripemd160_f(i, bl, cl, dl);
            const k = ripemd160_k_left(i);
            const r = ripemd160_r_left[i];
            const s = ripemd160_s_left[i];

            const temp = al +% f +% x[r] +% k;
            const rotated = rotl(temp, s);
            const t = rotated +% el;

            al = el;
            el = dl;
            dl = rotl(cl, 10);
            cl = bl;
            bl = t;
        }

        // Right rounds
        i = 0;
        while (i < 80) : (i += 1) {
            const f = ripemd160_f(79 - i, br, cr, dr);
            const k = ripemd160_k_right(i);
            const r = ripemd160_r_right[i];
            const s = ripemd160_s_right[i];

            const temp = ar +% f +% x[r] +% k;
            const rotated = rotl(temp, s);
            const t = rotated +% er;

            ar = er;
            er = dr;
            dr = rotl(cr, 10);
            cr = br;
            br = t;
        }

        // Update hash state
        const t = self.h[1] +% cl +% dr;
        self.h[1] = self.h[2] +% dl +% er;
        self.h[2] = self.h[3] +% el +% ar;
        self.h[3] = self.h[4] +% al +% br;
        self.h[4] = self.h[0] +% bl +% cr;
        self.h[0] = t;
    }
};

const ripemd160_initial_state = [5]u32{
    0x67452301,
    0xEFCDAB89,
    0x98BADCFE,
    0x10325476,
    0xC3D2E1F0,
};

fn ripemd160_f(j: usize, x: u32, y: u32, z: u32) u32 {
    if (j < 16) {
        return x ^ y ^ z;
    } else if (j < 32) {
        return (x & y) | (~x & z);
    } else if (j < 48) {
        return (x | ~y) ^ z;
    } else if (j < 64) {
        return (x & z) | (y & ~z);
    } else {
        return x ^ (y | ~z);
    }
}

fn ripemd160_k_left(j: usize) u32 {
    if (j < 16) return 0x00000000;
    if (j < 32) return 0x5A827999;
    if (j < 48) return 0x6ED9EBA1;
    if (j < 64) return 0x8F1BBCDC;
    return 0xA953FD4E;
}

fn ripemd160_k_right(j: usize) u32 {
    if (j < 16) return 0x50A28BE6;
    if (j < 32) return 0x5C4DD124;
    if (j < 48) return 0x6D703EF3;
    if (j < 64) return 0x7A6D76E9;
    return 0x00000000;
}

const ripemd160_r_left = [80]u8{
    0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    7, 4,  13, 1,  10, 6,  15, 3,  12, 0, 9,  5,  2,  14, 11, 8,
    3, 10, 14, 4,  9,  15, 8,  1,  2,  7, 0,  6,  13, 11, 5,  12,
    1, 9,  11, 10, 0,  8,  12, 4,  13, 3, 7,  15, 14, 5,  6,  2,
    4, 0,  5,  9,  7,  12, 2,  10, 14, 1, 3,  8,  11, 6,  15, 13,
};

const ripemd160_r_right = [80]u8{
    5,  14, 7,  0, 9, 2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12,
    6,  11, 3,  7, 0, 13, 5,  10, 14, 15, 8,  12, 4,  9,  1,  2,
    15, 5,  1,  3, 7, 14, 6,  9,  11, 8,  12, 2,  10, 0,  4,  13,
    8,  6,  4,  1, 3, 11, 15, 0,  5,  12, 2,  13, 9,  7,  10, 14,
    12, 15, 10, 4, 1, 5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11,
};

const ripemd160_s_left = [80]u5{
    11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,
    7,  6,  8,  13, 11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12,
    11, 13, 6,  7,  14, 9,  13, 15, 14, 8,  13, 6,  5,  12, 7,  5,
    11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,  8,  6,  5,  12,
    9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6,
};

const ripemd160_s_right = [80]u5{
    8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,
    9,  13, 15, 7,  12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11,
    9,  7,  15, 11, 8,  6,  6,  14, 12, 13, 5,  14, 13, 13, 7,  5,
    15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,  12, 5,  15, 8,
    8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11,
};

pub fn ripemd_160(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const input = args.value.unwrapBytestring();
    var hasher = RipeMd160Ctx.init();

    var chunk: [64]u8 = undefined;
    var chunk_len: usize = 0;
    const byte_len: usize = @intCast(input.length);

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        chunk[chunk_len] = @as(u8, @truncate(input.bytes[i]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }

    if (chunk_len != 0) {
        hasher.updateSlice(chunk[0..chunk_len]);
    }

    var digest: [20]u8 = undefined;
    hasher.finalize(&digest);

    const digest_words: u32 = 20;
    const allocation = builtins_bytestring.initByteStringAllocation(m.heap, digest_words);
    for (digest, 0..) |byte, idx| {
        allocation.data_words[idx] = byte;
    }

    return createConst(m.heap, @ptrCast(allocation.constant_words));
}
