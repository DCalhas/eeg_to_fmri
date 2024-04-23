import itertools
import pickle

import jpype
import jpype.imports
from jpype.types import *

import time, re
import pandas as pd

import os

DEFAULT_PARAMS={
'symmetries': False,
'normalization': "column",
'discretization': "normal_distribution",
'noise_relaxation': "optional",
'filling_criteria': "remove",
'pattern_type': "constant",
'orientation': "rows",
'remove_percentage': 0.4, # noise
'nr_iterations': 1,
'min_biclusters': 100,
'min_lift': 1.2,
'nr_labels': 5,
'min_columns': 3,
}

MAX_HEAP_SIZE_MB = 6 * 1024
print(f"Maximum heap size: {MAX_HEAP_SIZE_MB} MB")

if not jpype.isJVMStarted():
    jpype.startJVM(f"-Xmx{MAX_HEAP_SIZE_MB}M", classpath=[os.environ['BICPY_DIR']+'src/bicpams_5.jar'])  # TODO: fix hardcode

# Xmx flag -> maximum memory allocation, -Xmx8g = 8 Gb, default value is 246 MB
# Xms flag -> initial memory allocation, -Xms2048m = 2048 MB, no default value

from java.lang import String, Runtime, System
from java.io import PrintStream, File

from utils import BicReader, BicResult
from java.io import File
from domain import Dataset, Bicluster, Biclusters
from generator.BicMatrixGenerator import PatternType
from bicpam.bicminer.BiclusterMiner import Orientation
from bicpam.mapping import Itemizer
from bicpam.mapping.Itemizer import DiscretizationCriteria, FillingCriteria, NoiseRelaxation, NormalizationCriteria
from bicpam.closing import Biclusterizer
from bicpam.pminer.fim import ClosedFIM
from bicpam.pminer.spm import SequentialPM
from bicpam.pminer.spm.SequentialPM import SequentialImplementation
from bicpam.bicminer.coherent import AdditiveBiclusterMiner, MultiplicativeBiclusterMiner, SymmetricBiclusterMiner
from bicpam.bicminer.constant import ConstantBiclusterMiner, ConstantOverallBiclusterMiner
from bicpam.bicminer.order import OrderPreservingBiclusterMiner
from utils.others import CopyUtils
from performance.significance import BSignificance
import bicpam.bicminer


########## Pattern class to hold pattern information #########
class Pattern:
    """
        A class used to represent a Pattern returned by the biclustering algorithm

        Parameters
        ----------
        columns : list
            a list of strings with the column names of the pattern
        rows : list
            a list of strings with the row names/indexes of the pattern
        values : list
            a list of numbers with the values corresponding to each column of the pattern
        pvalue : float
            the p-value associated with the pattern
        lift : float
            the lift associated with the pattern
    """

    def __init__(self, columns: list, rows: list, values: list, pvalue: float, lift: float):
        self.columns = columns
        self.rows = rows
        self.values = values
        self.pvalue = pvalue
        self.lift = lift

    @staticmethod
    def parser(text: str) -> dict:
        """Parses a text returned by the biclustering and returns the Pattern's attributes in a dictionary"""

        properties = {}
        properties["columns"] = re.findall("Y=\[(.*?)\]", text)[0].split(",")
        properties["rows"] = re.findall("X=\[(.*?)\]", text)[0].split(",")
        temp = re.findall("I=\[(.*?)\]", text)[0].split(",")
        properties["values"] = list(map(int, temp))
        properties["pvalue"] = float(re.findall("pvalue=(.*?)\s", text)[0])
        properties["lift"] = float(re.findall("Lifts=\[(.*?)\]", text)[0].split(",")[0])  # gets first lift value
        return properties

    def __str__(self):
        string = ""
        string += f"Columns: {self.columns}\n"
        string += f"Rows: {self.rows}\n"
        string += f"Values: {self.values}\n"
        string += f"p-value: {self.pvalue}\n"
        string += f"Lift: {self.lift}\n"
        return string

    def __repr__(self):
        return f"Pattern(columns={self.columns}, rows={self.rows}, values={self.values}, pvalue={self.pvalue}, lift={self.lift})"


########## Functions to translate string key to java object ##########

##### Super Function ################

def get_super(key: str, dictionary: dict):
    """Super function for transforming a string (key) into a java Object using a dict (dictionary)"""

    key = key.lower()
    try:
        res = dictionary[key]
    except KeyError:
        raise ValueError(f"Argument '{key}' is invalid, must be one of the following: {list(dictionary.keys())}")
    return res


###################################

def get_orientation(key: str) -> Orientation:
    """Receives key and returns corresponding Orientation Java object"""

    orientation_dict = {
        "rows": Orientation.PatternOnRows,
        "columns": Orientation.PatternOnColumns
    }
    return get_super(key, orientation_dict)


def get_bicminer(key: str) -> bicpam.bicminer:
    """Receives key and returns corresponding bicpam.bicminer Java object"""

    pattern_type_dict = {
        "additive": AdditiveBiclusterMiner,
        "constant": ConstantBiclusterMiner,
        "symmetric": SymmetricBiclusterMiner,
        "constant_overall": ConstantOverallBiclusterMiner,
        "multiplicative": MultiplicativeBiclusterMiner,
        "order_perserving": OrderPreservingBiclusterMiner
    }
    return get_super(key, pattern_type_dict)


def get_normalization_criteria(key: str) -> NormalizationCriteria:
    """Receives key and returns corresponding NormalizationCriteria Java object"""

    normalization_type_dict = {
        "column": NormalizationCriteria.Column
    }
    return get_super(key, normalization_type_dict)


def get_discretization_criteria(key: str) -> DiscretizationCriteria:
    """Receives key and returns corresponding DiscretizationCriteria Java object"""

    discretization_type_dict = {
        "normal_distribution": DiscretizationCriteria.NormalDist
    }
    return get_super(key, discretization_type_dict)


def get_noise_relaxation(key: str) -> NoiseRelaxation:
    """Receives key and returns corresponding NoiseRelaxation Java object"""

    noise_type_dict = {
        "optional": NoiseRelaxation.OptionalItem
    }
    return get_super(key, noise_type_dict)


def get_filling_criteria(key: str) -> FillingCriteria:
    """Receives key and returns corresponding FillingCriteria Java object"""

    filling_type_dict = {
        "remove": FillingCriteria.RemoveValue
    }
    return get_super(key, filling_type_dict)


########## Functions to execute biclustering #############################
def read_dataset(path: str, class_index=0) -> Dataset:
    """Read java object domain.Dataset from given path, sets class according to class_index (default -1)"""

    # path = File("").getAbsolutePath() + path
    path = String(path)
    if path.contains(".arff"):
        data = Dataset(BicReader.getInstances(path), class_index)
    else:
        data = Dataset(BicReader.getConds(path, 1, ","), BicReader.getGenes(path, ","),
                       BicReader.getTable(path, 1, ","), class_index)

    return data


def itemizer(data: Dataset, nr_labels: int, symmetries: bool, normalization: str, discretization: str,
             noise_relaxation: str,
             filling_criteria: str) -> Dataset:
    """Discretizes numeric dataset according to arguments"""
    nr_labels = int(nr_labels)
    normalization = get_normalization_criteria(normalization)
    discretization = get_discretization_criteria(discretization)
    noise_relaxation = get_noise_relaxation(noise_relaxation)
    filling_criteria = get_filling_criteria(filling_criteria)

    data = Itemizer.run(data, nr_labels, symmetries,
                        normalization,
                        discretization,
                        noise_relaxation,  # multi-item assignments
                        filling_criteria)
    return data


def get_pminer(data: Dataset, pattern_type: str, orientation: str, min_biclusters: int, min_columns: int,
               min_lift: float) -> bicpam.bicminer:
    """Creates pattern miner according to arguments"""

    posthandler = Biclusterizer()
    miner = get_bicminer(pattern_type)
    orientation = get_orientation(orientation)

    if pattern_type == "order_perserving":
        pminer = SequentialPM()
        pminer.algorithm = SequentialImplementation.PrefixSpan
    else:
        pminer = ClosedFIM()

    pminer.inputMinNrBics(min_biclusters)
    pminer.inputMinColumns(min_columns)
    pminer.setClass(data.classValues, min_lift)
    # pminer.setTargetClass(targetClass,classSuperiority);
    bicminer = miner(data, pminer, posthandler, orientation)
    return bicminer


def run_bicpam(data: Dataset, bicminer: bicpam.bicminer, nr_iterations: int, orientation: str,
               remove_percentage: float, verbose=0) -> None:
    """Runs the biclustering algorithm, applying {bicminer} to the {data} for {nr_iterations} iterations

    The patterns returned by the biclustering are saved to the file '/output/result.txt'
    """

    orientation = get_orientation(orientation)
    duration = time.time()
    bics = Biclusters()
    originalIndexes = CopyUtils.copyIntList(data.indexes)
    originalScores = CopyUtils.copyIntList(data.intscores)

    if verbose > 0: print("### Running biclustering")
    for i in range(0, nr_iterations):
        if verbose > 0:
            print(f"\n## Iteration {i + 1} out of {nr_iterations}")
            print(f"Current heap size: {Runtime.getRuntime().totalMemory() / (1024 ** 2)} MB")
            print("# Mining biclusters")
        iBics = bicminer.mineBiclusters()
        if verbose > 0: print("# Removing")
        data.remove(iBics.getElementCounts(), remove_percentage, 1)
        bicminer.setData(data)
        bics.addAll(iBics)

    data.indexes = originalIndexes
    data.intscores = originalScores
    bics.computePatterns(data, orientation)
    BSignificance.run(data, bics)
    bics.orderPValue()

    duration = time.time() - duration
    if verbose > 0: print(f"Time: {duration} s")
    BicResult.println("FOUND BICS:" + str(bics.toString(data.rows, data.columns)))
    for bic in bics.getBiclusters():
        BicResult.println(bic.toString(data) + "\n\n")


def get_patterns() -> list:
    """Reads the patterns returned by the biclustering from a file and turns them into a Pattern object"""

    patterns_list = []

    output_file=os.environ['OUTPUT_DIR']+"result.txt"
    with open(output_file, "r") as file:
        output = file.read()

    patterns = output.split('\n\n')[0].split('\n')[1:]
    for p in patterns:
        args = Pattern.parser(p)
        pattern = Pattern(args["columns"], args["rows"], args["values"], args["pvalue"], args["lift"])
        patterns_list.append(pattern)

    return patterns_list


def run(params: dict, data_file: str, verbose=1) -> dict:
    """
    bicpy.run(bicpy.DEFAULT_PARAMS, "/path/to/datafile")
    """
    if verbose > 0: print(f"Maximum heap memory of JVM: {Runtime.getRuntime().maxMemory() / (1024 ** 2)} MB")
    original_out = System.out

    try:
        if verbose == 0: System.setOut(PrintStream(File("/dev/null"))) # NUL for windows, /dev/null for unix

        data = read_dataset(data_file)
        discrete_data = itemizer(data, params["nr_labels"], params["symmetries"], params["normalization"],
                                 params["discretization"], params["noise_relaxation"], params["filling_criteria"])

        bicminer = get_pminer(discrete_data, params["pattern_type"], params["orientation"], params["min_biclusters"],
                              params["min_columns"], params["min_lift"])
        run_bicpam(discrete_data, bicminer, params["nr_iterations"], params["orientation"], params["remove_percentage"],
                   verbose)
    finally:
        System.setOut(original_out)

    return 0


# Auxiliary Function #########################################################

def parse_discrete_dataset(text: str) -> pd.DataFrame:
    """ Transform string representation in a DataFrame

    Receives a string as returned by the method .toString() of object domain.Dataset and returns the corresponding
    pd.DataFrame
    """

    text = str(text).replace(' ', '').strip()

    columns = re.findall("Courses:\[(.*?)\]", text)[0].split(",")

    rows = []
    indexes = []
    for line in text.split('\n')[2:]:  # skip first 2 lines corresponding to listing of rows and columns
        split_line = line.strip().split("=>")
        index = split_line[0]
        values = split_line[1].split('|')
        values = list(filter(None, values))  # remove empty strings from list
        values = list(map(float, values))  # transfrom from string to floar
        rows.append(values)
        indexes.append(index)

    df = pd.DataFrame(rows, columns=columns, index=indexes)
    return df


def discretize_data(data_path: str, parameterization: dict, verbose=1) -> pd.DataFrame:
    """Discretizes data available at {data_path} using bicpy.itemizer

    In the process removes the target variable. It is not important since this auxiliary dataset is used to calculate
    new variables without altering the target
    """

    original_out = System.out
    try:
        if verbose == 0: System.setOut(PrintStream(File("/dev/null"))) # NUL for windows, /dev/null for unix

        data = read_dataset(data_path)

        discrete_data = itemizer(data, parameterization["nr_labels"], parameterization["symmetries"],
                                 parameterization["normalization"], parameterization["discretization"],
                                 parameterization["noise_relaxation"], parameterization["filling_criteria"])
        discrete_data_text = discrete_data.toString(False)
        df = parse_discrete_dataset(discrete_data_text)
    finally:
        System.setOut(original_out)

    return df


def create_parameterizations(p: dict) -> list:
    """Receives a dict with lists as values and returns list of dictionaries with possible combinations of these vals"""

    keys = []
    vals = []
    for key, item in p.items():
        if isinstance(item, list):
            keys.append(key)
            vals.append(item)
    combinations = list(itertools.product(*vals))

    parameterizations = []
    for comb in combinations:
        d = p.copy()
        for key, value in zip(keys, comb):
            d[key] = value
        parameterizations.append(d)
    return parameterizations
