namespace LionClap.Tests;

/// <summary>
/// A small reader of the maps in the service's <c>openapi.yaml</c>: it follows indentation, ignores comments, list items and block scalars, and returns every key path
/// with its scalar value. It exists so the contract test needs no YAML library; it relies on the layout convention written at the top of that file.
/// </summary>
public static class YamlOutline
{
    public static Dictionary<string, string> Read(string yaml)
    {
        var result = new Dictionary<string, string>();
        var stack = new List<(int Indent, string Key)>();
        int? skipBelow = null; // inside a block scalar: ignore everything indented deeper than the key that opened it
        foreach (var raw in yaml.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            var trimmed = line.TrimStart();
            if (trimmed.Length == 0 || trimmed.StartsWith('#')) continue;
            var indent = line.Length - trimmed.Length;
            if (skipBelow is { } s)
            {
                if (indent > s) continue;
                skipBelow = null;
            }
            if (trimmed.StartsWith("- ") || trimmed == "-") continue;

            var colon = trimmed.IndexOf(':');
            if (colon <= 0) continue;
            var key = trimmed[..colon].Trim().Trim('\'', '"');
            var value = trimmed[(colon + 1)..].Trim();

            while (stack.Count > 0 && stack[^1].Indent >= indent) stack.RemoveAt(stack.Count - 1);
            stack.Add((indent, key));
            var path = string.Join('.', stack.Select(e => e.Key));
            if (value is "|" or ">") { skipBelow = indent; result[path] = ""; continue; }
            result[path] = value.Trim('\'', '"');
        }
        return result;
    }
}
