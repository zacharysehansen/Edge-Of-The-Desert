// Stand-in for onnxruntime-web, so frontend/models.js can be imported under Node.
// The browser resolves that bare specifier through the importmap in index.html; Node
// has no such mapping, and installing the real package only to never run inference
// would add a 30 MB dependency to a check that is about module loading, not models.
export const env = { wasm: { wasmPaths: '' } };
export class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}
export const InferenceSession = {
    create: async () => ({
        inputNames: ['features'],
        outputNames: ['output'],
        run: async () => ({ output: { data: new Float32Array([0]) } }),
    }),
};
