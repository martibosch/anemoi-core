# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import ValidationError

from anemoi.models.schemas.data_processor import ImputerSchema
from anemoi.models.schemas.data_processor import NormalizerSchema
from anemoi.models.schemas.data_processor import PostprocessorSchema
from anemoi.models.schemas.data_processor import PreprocessorSchema
from anemoi.models.schemas.data_processor import PreprocessorTarget
from anemoi.models.schemas.data_processor import RemapperSchema


def test_preprocessor_with_raw_dict():
    raw_config = {"default": "mean-std", "min-max": ["x"], "max": ["y"], "none": ["z"], "mean-std": ["q"]}
    schema = PreprocessorSchema(_target_="anemoi.models.preprocessing.normalizer.InputNormalizer", config=raw_config)

    assert schema.target_ == "anemoi.models.preprocessing.normalizer.InputNormalizer"
    assert schema.config == raw_config


def test_preprocessor_with_normalizer_instance():
    normalizer_instance = NormalizerSchema(default="std", remap={"x": "z", "y": "x"})
    schema = PreprocessorSchema(
        _target_="anemoi.models.preprocessing.normalizer.InputNormalizer", config=normalizer_instance
    )

    assert schema.target_ == "anemoi.models.preprocessing.normalizer.InputNormalizer"
    assert isinstance(schema.config, NormalizerSchema)
    assert schema.config.default == "std"
    assert schema.config.remap == {"x": "z", "y": "x"}


def test_preprocessor_with_imputer_dict():
    raw_config = {"default": "none", "maximum": ["x"], "none": ["z"], "minimum": ["q"]}
    schema = PreprocessorSchema(_target_=PreprocessorTarget.imputer, config=raw_config)
    assert schema.target_ == PreprocessorTarget.imputer
    assert schema.config["default"] == "none"


def test_preprocessor_with_imputer_instance():
    instance = ImputerSchema(default="none", maximum=["x"], minimum=["q"], none=["z"])
    schema = PreprocessorSchema(_target_=PreprocessorTarget.imputer, config=instance)
    assert isinstance(schema.config, ImputerSchema)
    assert schema.config.maximum == ["x"]


def test_preprocessor_with_constant_imputer_dict():
    raw_config = {"default": 1.0, 1.0: ["x"], 5.0: ["y"], "none": ["z"]}
    schema = PreprocessorSchema(_target_=PreprocessorTarget.const_imputer, config=raw_config)
    assert schema.target_ == PreprocessorTarget.const_imputer
    assert schema.config["default"] == 1.0
    assert schema.config[1.0] == ["x"]


def test_preprocessor_with_postprocessor_dict():
    raw_config = {"default": "hardtanh_0_1", "hardtanh_0_1": ["x"], "none": ["y"]}
    schema = PreprocessorSchema(_target_=PreprocessorTarget.postprocessor, config=raw_config)
    assert schema.config["default"] == "hardtanh_0_1"


def test_preprocessor_with_postprocessor_instance():
    instance = PostprocessorSchema(default="relu", relu=["x"], none=["z"])
    schema = PreprocessorSchema(_target_=PreprocessorTarget.postprocessor, config=instance)
    assert isinstance(schema.config, PostprocessorSchema)
    assert schema.config.relu == ["x"]


def test_preprocessor_with_conditional_zero_postprocessor_dict():
    raw_config = {"default": 0.0, "remap": "ref_var", 0: ["x"], 1: ["y"]}
    schema = PreprocessorSchema(_target_=PreprocessorTarget.conditional_zero_postprocessor, config=raw_config)
    assert schema.config["remap"] == "ref_var"


def test_preprocessor_with_normalized_relu_postprocessor_dict():
    raw_config = {"normalizer": "mean-std", 1.0: ["x"], 0: ["y"]}
    schema = PreprocessorSchema(_target_=PreprocessorTarget.normalized_relu_postprocessor, config=raw_config)
    print(schema)
    assert schema.config["normalizer"] == "mean-std"


def test_preprocessor_with_remapper_instance():
    instance = RemapperSchema(default="log1p", none=["d", "q"])
    schema = PreprocessorSchema(_target_=PreprocessorTarget.remapper, config=instance)
    assert isinstance(schema.config, RemapperSchema)
    assert "d" in schema.config.none

def test_remapper_schema_covers_every_supported_method():
    """The schema must stay in sync with Remapper.supported_methods.

    BaseModel forbids extra keys, so a method implemented but not declared here is
    unusable from any validated config. This is the regression that let log1p/sqrt/
    boxcox be the only reachable methods while nine were implemented.
    """
    from anemoi.models.preprocessing.remapper import Remapper

    declared = set(RemapperSchema.model_fields)
    literals = set(RemapperSchema.model_fields["default"].json_schema_extra["literals"])

    for method in Remapper.supported_methods:
        assert method in declared, f"{method!r} is implemented but not declared in RemapperSchema"
        assert method in literals, f"{method!r} is not accepted as a RemapperSchema default"


def test_remapper_schema_accepts_method_kwargs():
    instance = RemapperSchema(default="none", asinh=["CAPE_ML"], method_kwargs={"asinh": {"c": 0.004}})
    assert instance.asinh == ["CAPE_ML"]
    assert instance.method_kwargs == {"asinh": {"c": 0.004}}


def test_remapper_schema_accepts_affine_with_scale_and_shift():
    instance = RemapperSchema(
        default="none",
        affine=["x"],
        method_kwargs={"affine": {"scale": 0.004, "shift": 0.0}},
    )
    schema = PreprocessorSchema(_target_=PreprocessorTarget.remapper, config=instance)
    assert isinstance(schema.config, RemapperSchema)
    assert schema.config.method_kwargs["affine"]["scale"] == 0.004


def test_remapper_schema_rejects_unknown_key():
    with pytest.raises(ValidationError):
        RemapperSchema(default="none", not_a_method=["x"])
