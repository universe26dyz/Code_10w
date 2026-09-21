function hex = multimap_sha256(bytes)
% Stable SHA256 used only for cache identity/provenance.
    if ischar(bytes) || isstring(bytes), bytes = unicode2native(char(bytes), 'UTF-8'); end
    engine = java.security.MessageDigest.getInstance('SHA-256');
    engine.update(uint8(bytes));
    raw = typecast(engine.digest(), 'uint8');
    hex = lower(reshape(dec2hex(raw, 2).', 1, []));
end
