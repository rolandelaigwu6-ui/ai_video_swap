"""Isolated, single-model CPU workers. No model remains resident between stages."""
import os
import sys
import resource


def main():
    # Virtual-address safety ceiling; parent also watches resident memory.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    import numpy as np
    mode, model, directory = sys.argv[1:]
    from pathlib import Path
    folder = Path(directory)
    if mode == 'map':
        import onnx
        from onnx import numpy_helper
        graph = onnx.load(model)
        matrix = numpy_helper.to_array(graph.graph.initializer[-1]).astype(np.float32)
        if matrix.shape != (512, 512):
            raise ValueError(f'Unexpected identity mapping shape: {matrix.shape}')
        np.save(folder / 'mapping.npy', matrix)
    else:
        import onnxruntime as ort
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(model, options, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        if mode == 'embed':
            image = np.load(folder / 'source.npy', allow_pickle=False)
            embedding = session.run(None, {inputs[0].name: image})[0].astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(embedding)
            if embedding.shape != (1, 512) or not np.isfinite(embedding).all() or norm < 1e-8:
                raise ValueError('Invalid source identity embedding')
            np.save(folder / 'embedding.npy', embedding / norm)
        elif mode in ('parse', 'occlude'):
            image = np.load(folder / 'mask-input.npy', allow_pickle=False)
            output = session.run(None, {inputs[0].name: image})[0]
            if mode == 'parse':
                if output.ndim != 4 or output.shape[1] != 19:
                    raise ValueError(f'Invalid face parser output: {output.shape}')
                output = output[0].argmax(axis=0).astype(np.uint8)
            else:
                output = output.squeeze().astype(np.float32)
                if output.ndim != 2 or not np.isfinite(output).all():
                    raise ValueError('Invalid occlusion output')
                output = output.clip(0,1)
            np.save(folder / (mode + '.npy'), output)
        elif mode in ('swap', 'hyperswap'):
            image = np.load(folder / 'target.npy', allow_pickle=False)
            latent = np.load(folder / 'embedding.npy', allow_pickle=False)
            if mode == 'swap':
                latent = latent @ np.load(folder / 'mapping.npy', allow_pickle=False)
            latent /= max(float(np.linalg.norm(latent)), 1e-8)
            feeds = {}
            for item in inputs:
                value = image if len(item.shape) == 4 else latent
                feeds[item.name] = value.astype(np.float16 if item.type == 'tensor(float16)' else np.float32)
            prediction = session.run(None, feeds)[0].astype(np.float32)
            expected = 256 if mode == 'hyperswap' else 128
            if prediction.shape != (1, 3, expected, expected) or not np.isfinite(prediction).all():
                raise ValueError('Invalid neural swap output')
            if mode == 'hyperswap':
                prediction = prediction * .5 + .5
            np.save(folder / 'prediction.npy', prediction)
        else:
            raise ValueError('Unknown worker stage')
    print(f'peak_rss_mib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}', flush=True)

if __name__ == '__main__':
    main()
