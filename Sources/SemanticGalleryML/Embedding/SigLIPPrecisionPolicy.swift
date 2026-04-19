import MLX

public enum SigLIPPrecisionPolicy {
    public static let deployment: DType = .bfloat16
    public static let training: DType = .bfloat16
}
