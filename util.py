from pathlib import Path
import pickle


# decorator to save the output of the function
def save_output(func):
    def wrapper(*args, output_path=None, override=False, load=False, **kwargs):
        if output_path is not None:
            if Path(output_path).exists():
                if load:
                    print(f'Loading output from {output_path}')
                    with open(output_path, 'rb') as f:
                        return pickle.load(f)
                if not override:
                    raise ValueError(f'Output file {output_path} already exists. Use --override to overwrite it.')

            output = func(*args, **kwargs)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)
            return output
        else:
            return func(*args, **kwargs)

    return wrapper
