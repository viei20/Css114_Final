import cmath
import math

def clean_val(v, tol=1e-5):
    if isinstance(v, complex):
        real = 0.0 if abs(v.real) < tol else v.real
        imag = 0.0 if abs(v.imag) < tol else v.imag
        if imag == 0: return round(real, 4)
        return complex(round(real, 4), round(imag, 4))
    else:
        return 0.0 if abs(v) < tol else round(v, 4)

def print_matrix(name, M):
    print(f"{name} =")
    for row in M:
        formatted_row = []
        for val in row:
            cv = clean_val(val)
            if isinstance(cv, complex):
                sign = "+" if cv.imag >= 0 else "-"
                formatted_row.append(f"{cv.real:8.4f}{sign}{abs(cv.imag):.4f}j")
            else:
                formatted_row.append(f"{cv:8.4f}")
        print("  [" + ", ".join(formatted_row) + "]")
    print()

def get_eigenvalues_2x2(A):
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    tr = a + d
    det = a*d - b*c
    
    discriminant = tr**2 - 4*1*det
    if abs(discriminant) < 1e-9: discriminant = 0
    
    if discriminant >= 0:
        l1 = (tr + discriminant**0.5) / 2
        l2 = (tr - discriminant**0.5) / 2
        return [l1, l2]
    else:
        d_imag = (-discriminant)**0.5
        l1 = complex(tr/2, d_imag/2)
        l2 = complex(tr/2, -d_imag/2)
        return [l1, l2]

def get_eigenvalues_3x3(A):

    a_11, a_12, a_13 = A[0]
    a_21, a_22, a_23 = A[1]
    a_31, a_32, a_33 = A[2]

    # หาสปส. Characteristic Equation: lambda^3 - tr(A)lambda^2 + c1*lambda - det(A) = 0
    tr = a_11 + a_22 + a_33
    
    # Sum of principal minors (c1)
    m1 = a_22*a_33 - a_23*a_32
    m2 = a_11*a_33 - a_13*a_31
    m3 = a_11*a_22 - a_12*a_21
    c1 = m1 + m2 + m3
    
    # Determinant
    det = (a_11 * (a_22*a_33 - a_23*a_32) -
           a_12 * (a_21*a_33 - a_23*a_31) +
           a_13 * (a_21*a_32 - a_22*a_31))

    # รูปแบบสมการ x^3 + bx^2 + cx + d = 0
    b = -tr
    c = c1
    d = -det
    
    p = c - (b**2)/3
    q = (2*b**3)/27 - (b*c)/3 + d
    
    delta = (q**2)/4 + (p**3)/27
    root_delta = cmath.sqrt(delta)
    
    u3 = -q/2 + root_delta
    v3 = -q/2 - root_delta

    def cube_roots(z):
        r = abs(z)**(1/3.0)
        theta = cmath.phase(z)/3.0
        return [
            r * cmath.rect(1, theta),
            r * cmath.rect(1, theta + 2*math.pi/3),
            r * cmath.rect(1, theta - 2*math.pi/3)
        ]

    u_roots = cube_roots(u3)
    v_roots = cube_roots(v3)

    roots = []
    for u in u_roots:
        for v in v_roots:
            if abs(u*v - (-p/3)) < 1e-5:
                roots.append(u + v - b/3)
                break
                
    clean_roots = []
    for r in roots:
        if abs(r.imag) < 1e-6: 
            clean_roots.append(r.real)
        else: 
            clean_roots.append(r)
            
    clean_roots.sort(key=lambda x: x.real if isinstance(x, complex) else x, reverse=True)
    return clean_roots

def rref(matrix):
    M = [row[:] for row in matrix]
    n = len(M)
    m = len(M[0])
    lead = 0
    for r in range(n):
        if lead >= m: break
        i = r
        while abs(M[i][lead]) < 1e-6:
            i += 1
            if i == n:
                i = r
                lead += 1
                if lead == m: return M
        
        M[i], M[r] = M[r], M[i]
        
        lv = M[r][lead]
        M[r] = [val / lv for val in M[r]]
        
        for i in range(n):
            if i != r:
                lv = M[i][lead]
                M[i] = [M[i][j] - lv * M[r][j] for j in range(m)]
        lead += 1
    return M

def find_null_space(M):
    n, m = len(M), len(M[0])
    pivot_rows, pivot_cols, free_cols = [], [], []
    
    for r in range(n):
        for c in range(m):
            if abs(M[r][c]) > 1e-6:
                pivot_rows.append(r)
                pivot_cols.append(c)
                break
                
    for c in range(m):
        if c not in pivot_cols:
            free_cols.append(c)
            
    null_space = []
    for f in free_cols:
        vec = [0] * m
        vec[f] = 1
        
        for idx in range(len(pivot_cols)):
            r = pivot_rows[idx]
            p_col = pivot_cols[idx]
            vec[p_col] = -M[r][f]
        null_space.append([clean_val(v) for v in vec])
        
    return null_space

def get_inverse(P):
    n = len(P)
    aug = [[P[i][j] for j in range(n)] + [(1 if i==j else 0) for j in range(n)] for i in range(n)]
    aug_rref = rref(aug)
    
    P_inv = []
    for i in range(n):
        P_inv.append(aug_rref[i][n:])
    return P_inv

def mat_mult(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0]*p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(m))
    return C

def solve_eigen_problem(A):
    n = len(A)
    print("\n" + "=" * 50)
    print_matrix("Input Matrix (A)", A)
    
    eigenvalues = get_eigenvalues_2x2(A) if n == 2 else get_eigenvalues_3x3(A)
    
    unique_vals, multiplicities = [], []
    for val in eigenvalues:
        found = False
        for i, u in enumerate(unique_vals):
            if abs(val - u) < 1e-5:
                multiplicities[i] += 1
                found = True
                break
        if not found:
            unique_vals.append(val)
            multiplicities.append(1)

    print("1. ค่าเจาะจง (Eigenvalues):")
    for i, val in enumerate(unique_vals):
        print(f"lambda = {clean_val(val)} (Algebraic Multiplicity = {multiplicities[i]})")
        
    print("\n2. ค่าเวกเตอร์เจาะจง (Eigenvectors):")
    P_columns = []
    is_diagonalizable = True
    
    for i, val in enumerate(unique_vals):
        M = [[A[r][c] - (val if r == c else 0) for c in range(n)] for r in range(n)]
        reduced_M = rref(M)
        basis = find_null_space(reduced_M)
        
        print(f"สำหรับ lambda = {clean_val(val)}:")
        if len(basis) == 0:
            print("  ไม่พบเวกเตอร์อิสระ")
        for j, vec in enumerate(basis):
            print(f"      v_{j+1} = {vec}")
            P_columns.append(vec)
            
        if len(basis) < multiplicities[i]:
            is_diagonalizable = False

    print("\n3. ตรวจสอบการแปลงเป็นเมทริกซ์ทแยงมุม (Diagonalizability):")
    if is_diagonalizable and len(P_columns) == n:
        print("  ✅ สามารถแปลงเป็นเมทริกซ์ทแยงมุมได้ (Diagonalizable)")
        print(f"  เหตุผล: มีเวกเตอร์เจาะจงที่เป็นอิสระเชิงเส้นครบ {n} ตัว \n")
        
        P = [[P_columns[c][r] for c in range(n)] for r in range(n)]
        P_inv = get_inverse(P)
        
        print("4. เมทริกซ์ P และ P^-1:")
        print_matrix("P (Eigenvector Matrix)", P)
        print_matrix("P^-1 (Inverse of P)", P_inv)
        
        AP = mat_mult(A, P)
        D_calc = mat_mult(P_inv, AP)
        print_matrix("D (พิสูจน์จากการคำนวณ P^-1 * A * P)", D_calc)
        
    else:
        print("  ❌ ไม่สามารถแปลงเป็นเมทริกซ์ทแยงมุมได้ (Not Diagonalizable)")
        print(f"  เหตุผล: จำนวนเวกเตอร์เจาะจงที่เป็นอิสระเชิงเส้นมีไม่ครบ {n} ตัว")
        print("  (Geometric Multiplicity < Algebraic Multiplicity)")
        print("  จึงไม่สามารถสร้างเมทริกซ์ P ที่มีอินเวอร์ส (P^-1) ได้")

def get_user_matrix():
    print("=== โปรแกรมคำนวณ Eigenvalue / Eigenvector และ Diagonalization ===")
    while True:
        try:
            n = int(input("กรุณาระบุขนาดของเมทริกซ์ (พิมพ์ 2 สำหรับ 2x2 หรือ 3 สำหรับ 3x3): "))
            if n not in [2, 3]:
                print("! โปรแกรมรองรับเฉพาะ 2x2 หรือ 3x3 เท่านั้น กรุณาลองใหม่อีกครั้ง !")
                continue
            break
        except ValueError:
            print("!! กรุณาป้อนตัวเลขจำนวนเต็ม !!")
    
    print(f"\nกรุณาป้อนสมาชิกของเมทริกซ์ขนาด {n}x{n} ทีละแถว")
    print("ตัวอย่างการป้อน: หากแถวแรกคือ 4, 0, 1 ให้พิมพ์ 4 0 1 แล้วกด Enter")
    
    matrix = []
    for i in range(n):
        while True:
            try:
                row_input = input(f"แถวที่ {i+1}: ")
                row = [float(x) for x in row_input.split()]
                if len(row) != n:
                    print(f"! คุณป้อนสมาชิกไม่ครบหรือเกิน {n} ตัว กรุณาป้อนใหม่ให้ถูกต้อง !")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("! กรุณาป้อนเฉพาะตัวเลข (คั่นด้วยช่องว่าง) เท่านั้น !")
    return matrix


if __name__ == "__main__":
    while True:
        user_matrix = get_user_matrix()
        solve_eigen_problem(user_matrix)
        
        cont = input("\nต้องการคำนวณเมทริกซ์อื่นอีกหรือไม่? (y/n): ").strip().lower()
        if cont != 'y':
            print("จบการทำงานของโปรแกรม")
            break
        print("\n" + "="*50 + "\n")
