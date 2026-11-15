using TorchSharp;
using static TorchSharp.torch;

namespace LionClap.Audio.Htsat;

/// <summary>Window geometry of the Swin blocks: partition/reverse, the relative position index and the shifted-window attention mask.</summary>
public static class SwinWindows
{
    public const float MaskValue = -100.0f;

    /// <summary><c>[B, H, W, C]</c> → <c>[B·(H/w)·(W/w), w, w, C]</c>, windows ordered by batch, then window row, then window column.</summary>
    public static Tensor Partition(Tensor x, int window)
    {
        var (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        if (h % window != 0 || w % window != 0)
            throw new ArgumentException($"a {h}×{w} map is not divisible into {window}×{window} windows", nameof(x));
        using var split = x.reshape(b, h / window, window, w / window, window, c);
        using var permuted = split.permute(0, 1, 3, 2, 4, 5).contiguous();
        return permuted.reshape(-1, window, window, c);
    }

    /// <summary>The inverse of <see cref="Partition"/>: <c>[B·(H/w)·(W/w), w, w, C]</c> → <c>[B, H, W, C]</c>.</summary>
    public static Tensor Reverse(Tensor windows, int window, long height, long width)
    {
        var c = windows.shape[^1];
        using var split = windows.reshape(-1, height / window, width / window, window, window, c);
        using var permuted = split.permute(0, 1, 3, 2, 4, 5).contiguous();
        return permuted.reshape(-1, height, width, c);
    }

    /// <summary>
    /// For every ordered pair of positions of a window, the row of the relative position bias table that applies:
    /// <c>(Δrow + w − 1)·(2w − 1) + (Δcol + w − 1)</c>. Shape <c>[w², w²]</c>, int64.
    /// </summary>
    public static Tensor RelativePositionIndex(int window)
    {
        var n = window * window;
        var index = new long[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var dr = i / window - j / window + window - 1;
                var dc = i % window - j % window + window - 1;
                index[i * n + j] = dr * (2L * window - 1) + dc;
            }
        return tensor(index, [n, n]);
    }

    /// <summary>
    /// Attention mask of the shifted blocks, <c>[windows, w², w²]</c> float32: 0 where two positions come from the same region of the
    /// 3×3 split of the (un-shifted) map and −100 where they do not.
    /// </summary>
    public static Tensor ShiftMask(int height, int width, int window, int shift)
    {
        if (shift <= 0 || shift >= window) throw new ArgumentOutOfRangeException(nameof(shift), shift, "a shifted block needs 0 < shift < window");
        if (height % window != 0 || width % window != 0)
            throw new ArgumentException($"a {height}×{width} map is not divisible into {window}×{window} windows");

        // region id of every pixel: slices [0, size−window), [size−window, size−shift), [size−shift, size)
        static int Slice(int p, int size, int window, int shift) => p < size - window ? 0 : p < size - shift ? 1 : 2;
        var n = window * window;
        var perRow = width / window;
        var windows = (height / window) * perRow;
        var mask = new float[windows * n * n];
        var ids = new int[n];
        for (var wr = 0; wr < height / window; wr++)
            for (var wc = 0; wc < perRow; wc++)
            {
                for (var i = 0; i < n; i++)
                {
                    var y = wr * window + i / window;
                    var x = wc * window + i % window;
                    ids[i] = Slice(y, height, window, shift) * 3 + Slice(x, width, window, shift);
                }
                var offset = (wr * perRow + wc) * n * n;
                for (var i = 0; i < n; i++)
                    for (var j = 0; j < n; j++)
                        mask[offset + i * n + j] = ids[i] == ids[j] ? 0.0f : MaskValue;
            }
        return tensor(mask, [windows, n, n]);
    }
}
