#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bpe_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_nanotok_cpp, m) {
    m.doc() = "High-performance BPE tokenizer engine";

    py::class_<BPEEngine::AddedToken>(m, "AddedToken")
        .def_readonly("content", &BPEEngine::AddedToken::content)
        .def_readonly("id", &BPEEngine::AddedToken::id)
        .def_readonly("special", &BPEEngine::AddedToken::special);

    py::class_<BPEEngine::Config>(m, "TokenizerConfig")
        .def(py::init<>())
        .def_readwrite("add_prefix_space", &BPEEngine::Config::add_prefix_space)
        .def_readwrite("trim_offsets", &BPEEngine::Config::trim_offsets);

    m.attr("PT_DEFAULT")   = static_cast<int>(BPEEngine::PT_DEFAULT);
    m.attr("PT_CL100K")   = static_cast<int>(BPEEngine::PT_CL100K);
    m.attr("PT_O200K")    = static_cast<int>(BPEEngine::PT_O200K);
    m.attr("PT_GPT2")     = static_cast<int>(BPEEngine::PT_GPT2);
    m.attr("PT_METASPACE") = static_cast<int>(BPEEngine::PT_METASPACE);
    m.attr("PT_SPLIT_SPC") = static_cast<int>(BPEEngine::PT_SPLIT_SPC);

    py::class_<BPEEngine>(m, "BPEEngine")
        .def(py::init<const std::string&>(), py::arg("tokenizer_json_path"))
        .def_static("from_json_string", &BPEEngine::from_json_string,
             py::arg("json_str"))
        .def("set_pretokenizer_mode", &BPEEngine::set_pretokenizer_mode,
             py::arg("mode"))
        .def("get_pretokenizer_mode", &BPEEngine::get_pretokenizer_mode)
        .def("encode", &BPEEngine::encode,
             py::arg("text"),
             py::arg("allowed_special") = std::unordered_set<std::string>())
        .def("batch_encode", &BPEEngine::batch_encode,
             py::arg("texts"),
             py::arg("allowed_special") = std::unordered_set<std::string>())
        .def("encode_chunks", &BPEEngine::encode_chunks,
             py::arg("chunks"))
        .def("decode", &BPEEngine::decode,
             py::arg("ids"))
        .def("batch_decode", &BPEEngine::batch_decode,
             py::arg("batch_ids"))
        .def("token_to_id", &BPEEngine::token_to_id,
             py::arg("token"))
        .def("vocab_size", &BPEEngine::vocab_size)
        .def("id_to_token", &BPEEngine::id_to_token,
             py::arg("id"))
        .def("get_added_tokens", &BPEEngine::get_added_tokens)
        .def("get_all_special_ids", &BPEEngine::get_all_special_ids)
        .def("config", &BPEEngine::config)
        .def("debug_classify", &BPEEngine::debug_classify, py::arg("codepoint"))
        .def("debug_letter_ranges_count", &BPEEngine::debug_letter_ranges_count)
        .def("debug_chunks", &BPEEngine::debug_chunks, py::arg("text"));
}
