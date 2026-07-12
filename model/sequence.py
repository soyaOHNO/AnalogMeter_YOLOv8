a = float(input("Enter the starting value (a1): "))
d = float(input("Enter the common difference (d): "))
n = int(input("Enter the number of terms (n): "))
r = float(input("Enter the first radian (r): "))

for i in range(n):
    an = a + d * i
    r = r - an
    print(f"Term {i + 1}: {r:.4f} radians")
    