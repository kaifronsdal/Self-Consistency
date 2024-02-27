from pathlib import Path
import pickle


# decorator to save the output of the function
def save_output(func):
    def wrapper(*args, output_path=None, override=False, load=False, **kwargs):
        if output_path is not None:
            if Path(output_path).exists():
                if not override:
                    raise ValueError(f'Output file {output_path} already exists. Use --override to overwrite it.')
                if load:
                    with open(output_path, 'rb') as f:
                        return pickle.load(f)

            output = func(*args, **kwargs)
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)
            return output
        else:
            return func(*args, **kwargs)

    return wrapper
