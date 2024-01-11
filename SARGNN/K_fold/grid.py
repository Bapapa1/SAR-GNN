
from pathlib import Path
import yaml
from copy import deepcopy


class grid_search:
    def __init__(self,dataset_name,hyper_file,args):
        self.args = args
        self.dataset_name=dataset_name
        self.hyper_file=hyper_file
        self.hyper_list=self._creat_grid()



    def _read_hyper_file(self):
        path = Path(self.hyper_file)
        if path.suffix in [".yaml", ".yml"]:
            if self.args.local_rank == 0:
                print('-----read model_hyper-----')
            return yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        else:
            raise ValueError

    def _grid_generator(self,hyper):
        keys = hyper.keys()
        results = {}

        if hyper == {}:
            yield {}
        else:
            configs_copy = deepcopy(hyper)  # create a copy to remove keys

            # get the "first" key
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = hyper[param]
            for value in first_key_values:
                results[param] = value

                for nested_config in self._grid_generator(configs_copy):
                    results.update(nested_config)
                    yield deepcopy(results)


    def _creat_grid(self):
        if self.args.local_rank == 0:
            print('----creat hyper-param grid search----')
        hyper=self._read_hyper_file()
        self.hyper_dict=hyper
        hyper_list=[cfg for cfg in self._grid_generator(hyper)]
        if self.args.local_rank == 0:
            print('----creat hyper-param grid search is finish----')
        return hyper_list




