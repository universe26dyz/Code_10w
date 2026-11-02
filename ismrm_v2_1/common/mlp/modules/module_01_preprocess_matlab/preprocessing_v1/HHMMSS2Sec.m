function time_sec = HHMMSS2Sec(HHMMSS)
numeric_val = str2double(HHMMSS);
HH = floor(numeric_val / 10000);
MM = floor(rem(numeric_val, 10000) / 100);
SS = floor(rem(numeric_val, 100));
frac_sec = numeric_val - floor(numeric_val);
time_sec = HH * 3600 + MM * 60 + SS + frac_sec;
end