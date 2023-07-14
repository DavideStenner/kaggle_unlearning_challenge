if __name__=='__main__':
    from src.utils import ignore_warning
    ignore_warning()

    from src.run_experiment import run_experiment
    
    run_experiment()