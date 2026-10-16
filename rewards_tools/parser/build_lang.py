from tree_sitter import Language, Parser

Language.build_library(
  'my-languages.so',

  [
    'tree_sitter_c',
    'tree_sitter_cpp',
  ]
)

