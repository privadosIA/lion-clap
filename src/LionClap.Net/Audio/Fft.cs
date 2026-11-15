namespace LionClap.Audio;

/// <summary>In-place iterative radix-2 complex FFT in double precision, with the twiddles and the bit reversal computed once. Immutable and thread-safe.</summary>
internal sealed class Fft
{
    private readonly int _n;
    private readonly double[] _cos;
    private readonly double[] _sin;
    private readonly int[] _reverse;

    public Fft(int n)
    {
        if (n < 2 || (n & (n - 1)) != 0) throw new ArgumentException("size must be a power of two", nameof(n));
        _n = n;
        _cos = new double[n / 2];
        _sin = new double[n / 2];
        for (var k = 0; k < n / 2; k++)
        {
            var angle = -2.0 * Math.PI * k / n;
            _cos[k] = Math.Cos(angle);
            _sin[k] = Math.Sin(angle);
        }
        _reverse = new int[n];
        var bits = System.Numerics.BitOperations.Log2((uint)n);
        for (var i = 0; i < n; i++)
        {
            var r = 0;
            for (var b = 0; b < bits; b++) if ((i & (1 << b)) != 0) r |= 1 << (bits - 1 - b);
            _reverse[i] = r;
        }
    }

    public int Size => _n;

    /// <summary>Forward transform (exp(-2πi·kn/N)) of <paramref name="re"/> + i·<paramref name="im"/>, in place.</summary>
    public void Transform(double[] re, double[] im)
    {
        for (var i = 0; i < _n; i++)
        {
            var j = _reverse[i];
            if (j > i)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        for (var size = 2; size <= _n; size <<= 1)
        {
            var half = size >> 1;
            var step = _n / size;
            for (var start = 0; start < _n; start += size)
            {
                for (var k = 0; k < half; k++)
                {
                    var wr = _cos[k * step];
                    var wi = _sin[k * step];
                    var a = start + k;
                    var b = a + half;
                    var tr = re[b] * wr - im[b] * wi;
                    var ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }
}
