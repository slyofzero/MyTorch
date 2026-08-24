import mytorch

def main():
    print("--- Testing PyTorch-from-Scratch (mytorch) ---")

    t1 = mytorch.Tensor([1, 2, 3, 4], [2, 2])
    t2 = mytorch.Tensor([10, 20, 30, 40], [2, 2])

    print("t1:", t1)
    print("t2:", t2)
    print("t1 shape:", t1.shape())
    print("t1 strides:", t1.strides())

    t3 = t1 + t2
    print("\nt3 = t1 + t2:", t3)
    print("t3[0, 0] =", t3[0, 0])
    print("t3[0, 1] =", t3[0, 1])
    print("t3[1, 0] =", t3[1, 0])
    print("t3[1, 1] =", t3[1, 1])

    t4 = t1 * t2
    print("\nt4 = t1 * t2:", t4)
    print("t4[1, 1] =", t4[1, 1])


if __name__ == "__main__":
    main()
