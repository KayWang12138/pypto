# GitHub 静态检查工具规则详细列表

**生成时间**: 2026-02-04

本文档列出了 GitHub pypto 项目中使用的所有静态检查工具的具体规则。

---

## 目录

1. [Ruff 规则 (Python)](#ruff-规则-python)
2. [cpplint 规则 (C/C++)](#cpplint-规则-cc)
3. [clang-format 配置 (C/C++)](#clang-format-配置-cc)
4. [Pyright 配置 (Python)](#pyright-配置-python)
5. [自定义检查规则](#自定义检查规则)

---

## Ruff 规则 (Python)

**总规则数**: 295 条

**启用的规则类别**:
- **PL (Pylint)**: 114 条 - Python 代码质量和风格检查
- **I (isort)**: 13 条 - Import 语句排序
- **E (pycodestyle errors)**: 69 条 - PEP 8 错误检查
- **W (pycodestyle warnings)**: 7 条 - PEP 8 警告检查
- **F (pyflakes)**: 92 条 - 逻辑错误检查

### PL: Pylint - 代码质量和风格

共 114 条规则

#### PLC0105 - type-name-incorrect-variance

`{kind}` name "{param_name}" does not reflect its {variance}; consider renaming it to "{replacement_name}"

#### PLC0131 - type-bivariance

`{kind}` cannot be both covariant and contravariant

#### PLC0132 - type-param-name-mismatch

`{kind}` name `{param_name}` does not match assigned variable name `{var_name}`

#### PLC0205 - single-string-slots

Class `__slots__` should be a non-string iterable

#### PLC0206 - dict-index-missing-items

Extracting value from dictionary without calling `.items()`

#### PLC0207 - missing-maxsplit-arg

String is split more times than necessary

#### PLC0208 - iteration-over-set

Use a sequence type instead of a `set` when iterating over values

#### PLC0414 - useless-import-alias

Import alias does not rename original package

#### PLC0415 - import-outside-top-level

`import` should be at the top-level of a file

#### PLC1802 - len-test

`len({expression})` used as condition without comparison

#### PLC1901 - compare-to-empty-string

`{existing}` can be simplified to `{replacement}` as an empty string is falsey

#### PLC2401 - non-ascii-name

{kind} name `{name}` contains a non-ASCII character

#### PLC2403 - non-ascii-import-name

Module alias `{name}` contains a non-ASCII character

#### PLC2701 - import-private-name

Private name import `{name}` from external module `{module}`

#### PLC2801 - unnecessary-dunder-call

Unnecessary dunder call to `{method}`. {replacement}.

#### PLC3002 - unnecessary-direct-lambda-call

Lambda expression called directly. Execute the expression inline instead.

#### PLE0100 - yield-in-init

`__init__` method is a generator

#### PLE0101 - return-in-init

Explicit return in `__init__`

#### PLE0115 - nonlocal-and-global

Name `{name}` is both `nonlocal` and `global`

#### PLE0116 - continue-in-finally

`continue` not supported inside `finally` clause

<details>
<summary>查看更多 94 条规则</summary>

#### PLE0117 - nonlocal-without-binding

Nonlocal name `{name}` found without binding

#### PLE0118 - load-before-global-declaration

Name `{name}` is used prior to global declaration on {row}

#### PLE0237 - non-slot-assignment

Attribute `{name}` is not defined in class's `__slots__`

#### PLE0241 - duplicate-bases

Duplicate base `{base}` for class `{class}`

#### PLE0302 - unexpected-special-method-signature

The special method `{}` expects {}, {} {} given

#### PLE0303 - invalid-length-return-type

`__len__` does not return a non-negative integer

#### PLE0304 - invalid-bool-return-type

`__bool__` does not return `bool`

#### PLE0305 - invalid-index-return-type

`__index__` does not return an integer

#### PLE0307 - invalid-str-return-type

`__str__` does not return `str`

#### PLE0308 - invalid-bytes-return-type

`__bytes__` does not return `bytes`

#### PLE0309 - invalid-hash-return-type

`__hash__` does not return an integer

#### PLE0604 - invalid-all-object

Invalid object in `__all__`, must contain only strings

#### PLE0605 - invalid-all-format

Invalid format for `__all__`, must be `tuple` or `list`

#### PLE0643 - potential-index-error

Expression is likely to raise `IndexError`

#### PLE0704 - misplaced-bare-raise

Bare `raise` statement is not inside an exception handler

#### PLE1132 - repeated-keyword-argument

Repeated keyword argument: `{duplicate_keyword}`

#### PLE1141 - dict-iter-missing-items

Unpacking a dictionary in iteration without calling `.items()`

#### PLE1142 - await-outside-async

`await` should be used within an async function

#### PLE1205 - logging-too-many-args

Too many arguments for `logging` format string

#### PLE1206 - logging-too-few-args

Not enough arguments for `logging` format string

#### PLE1300 - bad-string-format-character

Unsupported format character '{format_char}'

#### PLE1307 - bad-string-format-type

Format type does not match argument type

#### PLE1310 - bad-str-strip-call

String `{strip}` call contains duplicate characters (did you mean `{removal}`?)

#### PLE1507 - invalid-envvar-value

Invalid type for initial `os.getenv` argument; expected `str`

#### PLE1519 - singledispatch-method

`@singledispatch` decorator should not be used on methods

#### PLE1520 - singledispatchmethod-function

`@singledispatchmethod` decorator should not be used on non-method functions

#### PLE1700 - yield-from-in-async-function

`yield from` statement in async function; use `async for` instead

#### PLE2502 - bidirectional-unicode

Contains control characters that can permit obfuscated code

#### PLE2510 - invalid-character-backspace

Invalid unescaped character backspace, use "\b" instead

#### PLE2512 - invalid-character-sub

Invalid unescaped character SUB, use "\x1a" instead

#### PLE2513 - invalid-character-esc

Invalid unescaped character ESC, use "\x1b" instead

#### PLE2514 - invalid-character-nul

Invalid unescaped character NUL, use "\0" instead

#### PLE2515 - invalid-character-zero-width-space

Invalid unescaped character zero-width-space, use "\u200B" instead

#### PLE4703 - modified-iterating-set

Iterated set `{name}` is modified within the `for` loop

#### PLR0124 - comparison-with-itself

Name compared with itself, consider replacing `{actual}`

#### PLR0133 - comparison-of-constant

Two constants compared in a comparison, consider replacing `{left_constant} {op} {right_constant}`

#### PLR0202 - no-classmethod-decorator

Class method defined without decorator

#### PLR0203 - no-staticmethod-decorator

Static method defined without decorator

#### PLR0206 - property-with-parameters

Cannot have defined parameters for properties

#### PLR0402 - manual-from-import

Use `from {module} import {name}` in lieu of alias

#### PLR0904 - too-many-public-methods

Too many public methods ({methods} > {max_methods})

#### PLR0911 - too-many-return-statements

Too many return statements ({returns} > {max_returns})

#### PLR0912 - too-many-branches

Too many branches ({branches} > {max_branches})

#### PLR0913 - too-many-arguments

Too many arguments in function definition ({c_args} > {max_args})

#### PLR0914 - too-many-locals

Too many local variables ({current_amount}/{max_amount})

#### PLR0915 - too-many-statements

Too many statements ({statements} > {max_statements})

#### PLR0916 - too-many-boolean-expressions

Too many Boolean expressions ({expressions} > {max_expressions})

#### PLR0917 - too-many-positional-arguments

Too many positional arguments ({c_pos}/{max_pos})

#### PLR1701 - repeated-isinstance-calls

Merge `isinstance` calls: `{expression}`

#### PLR1702 - too-many-nested-blocks

Too many nested blocks ({nested_blocks} > {max_nested_blocks})

#### PLR1704 - redefined-argument-from-local

Redefining argument with the local name `{name}`

#### PLR1706 - and-or-ternary

Consider using if-else expression

#### PLR1708 - stop-iteration-return

Explicit `raise StopIteration` in generator

#### PLR1711 - useless-return

Useless `return` statement at end of function

#### PLR1714 - repeated-equality-comparison

Consider merging multiple comparisons: `{expression}`. Use a `set` if the elements are hashable.

#### PLR1716 - boolean-chained-comparison

Contains chained boolean comparison that can be simplified

#### PLR1722 - sys-exit-alias

Use `sys.exit()` instead of `{name}`

#### PLR1730 - if-stmt-min-max

Replace `if` statement with `{replacement}`

#### PLR1733 - unnecessary-dict-index-lookup

Unnecessary lookup of dictionary value by key

#### PLR1736 - unnecessary-list-index-lookup

List index lookup in `enumerate()` loop

#### PLR2004 - magic-value-comparison

Magic value used in comparison, consider replacing `{value}` with a constant variable

#### PLR2044 - empty-comment

Line with empty comment

#### PLR5501 - collapsible-else-if

Use `elif` instead of `else` then `if`, to reduce indentation

#### PLR6104 - non-augmented-assignment

Use `{operator}` to perform an augmented assignment directly

#### PLR6201 - literal-membership

Use a set literal when testing for membership

#### PLR6301 - no-self-use

Method `{method_name}` could be a function, class method, or static method

#### PLW0108 - unnecessary-lambda

Lambda may be unnecessary; consider inlining inner function

#### PLW0120 - useless-else-on-loop

`else` clause on loop without a `break` statement; remove the `else` and dedent its contents

#### PLW0127 - self-assigning-variable

Self-assignment of variable `{name}`

#### PLW0128 - redeclared-assigned-name

Redeclared variable `{name}` in assignment

#### PLW0129 - assert-on-string-literal

Asserting on an empty string literal will never pass

#### PLW0131 - named-expr-without-context

Named expression used without context

#### PLW0133 - useless-exception-statement

Missing `raise` statement on exception

#### PLW0177 - nan-comparison

Comparing against a NaN value; use `math.isnan` instead

#### PLW0211 - bad-staticmethod-argument

First argument of a static method should not be named `{argument_name}`

#### PLW0244 - redefined-slots-in-subclass

Slot `{slot_name}` redefined from base class `{base}`

#### PLW0245 - super-without-brackets

`super` call is missing parentheses

#### PLW0406 - import-self

Module `{name}` imports itself

#### PLW0602 - global-variable-not-assigned

Using global for `{name}` but no assignment is done

#### PLW0603 - global-statement

Using the global statement to update `{name}` is discouraged

#### PLW0604 - global-at-module-level

`global` at module level is redundant

#### PLW0642 - self-or-cls-assignment

Reassigned `{}` variable in {method_type} method

#### PLW0711 - binary-op-exception

Exception to catch is the result of a binary `and` operation

#### PLW1501 - bad-open-mode

`{mode}` is not a valid mode for `open`

#### PLW1507 - shallow-copy-environ

Shallow copy of `os.environ` via `copy.copy(os.environ)`

#### PLW1508 - invalid-envvar-default

Invalid type for environment variable default; expected `str` or `None`

#### PLW1509 - subprocess-popen-preexec-fn

`preexec_fn` argument is unsafe when using threads

#### PLW1510 - subprocess-run-without-check

`subprocess.run` without explicit `check` argument

#### PLW1514 - unspecified-encoding

`{function_name}` in text mode without explicit `encoding` argument

#### PLW1641 - eq-without-hash

Object does not implement `__hash__` method

#### PLW2101 - useless-with-lock

Threading lock directly created in `with` statement has no effect

#### PLW2901 - redefined-loop-name

Outer {outer_kind} variable `{name}` overwritten by inner {inner_kind} target

#### PLW3201 - bad-dunder-method-name

Dunder method `{name}` has no special meaning in Python 3

#### PLW3301 - nested-min-max

Nested `{func}` calls can be flattened

</details>

### I: isort - Import 排序

共 13 条规则

#### I001 - unsorted-imports

Import block is un-sorted or un-formatted

#### I002 - missing-required-import

Missing required import: `{name}`

#### ICN001 - unconventional-import-alias

`{name}` should be imported as `{asname}`

#### ICN002 - banned-import-alias

`{name}` should not be imported as `{asname}`

#### ICN003 - banned-import-from

Members of `{name}` should not be imported explicitly

#### INP001 - implicit-namespace-package

File `{filename}` is part of an implicit namespace package. Add an `__init__.py`.

#### INT001 - f-string-in-get-text-func-call

f-string is resolved before function call; consider `_("string %s") % arg`

#### INT002 - format-in-get-text-func-call

`format` method argument is resolved before function call; consider `_("string %s") % arg`

#### INT003 - printf-in-get-text-func-call

printf-style format is resolved before function call; consider `_("string %s") % arg`

#### ISC001 - single-line-implicit-string-concatenation

Implicitly concatenated string literals on one line

#### ISC002 - multi-line-implicit-string-concatenation

Implicitly concatenated string literals over multiple lines

#### ISC003 - explicit-string-concatenation

Explicitly concatenated string should be implicitly concatenated

#### ISC004 - implicit-string-concatenation-in-collection-literal

Unparenthesized implicit string concatenation in collection

### E: pycodestyle - 错误检查

共 69 条规则

#### E101 - mixed-spaces-and-tabs

Indentation contains mixed spaces and tabs

#### E111 - indentation-with-invalid-multiple

Indentation is not a multiple of {indent_width}

#### E112 - no-indented-block

Expected an indented block

#### E113 - unexpected-indentation

Unexpected indentation

#### E114 - indentation-with-invalid-multiple-comment

Indentation is not a multiple of {indent_width} (comment)

#### E115 - no-indented-block-comment

Expected an indented block (comment)

#### E116 - unexpected-indentation-comment

Unexpected indentation (comment)

#### E117 - over-indented

Over-indented (comment)

#### E201 - whitespace-after-open-bracket

Whitespace after '{symbol}'

#### E202 - whitespace-before-close-bracket

Whitespace before '{symbol}'

#### E203 - whitespace-before-punctuation

Whitespace before '{symbol}'

#### E204 - whitespace-after-decorator

Whitespace after decorator

#### E211 - whitespace-before-parameters

Whitespace before '{bracket}'

#### E221 - multiple-spaces-before-operator

Multiple spaces before operator

#### E222 - multiple-spaces-after-operator

Multiple spaces after operator

#### E223 - tab-before-operator

Tab before operator

#### E224 - tab-after-operator

Tab after operator

#### E225 - missing-whitespace-around-operator

Missing whitespace around operator

#### E226 - missing-whitespace-around-arithmetic-operator

Missing whitespace around arithmetic operator

#### E227 - missing-whitespace-around-bitwise-or-shift-operator

Missing whitespace around bitwise or shift operator

<details>
<summary>查看更多 49 条规则</summary>

#### E228 - missing-whitespace-around-modulo-operator

Missing whitespace around modulo operator

#### E231 - missing-whitespace

Missing whitespace after {}

#### E241 - multiple-spaces-after-comma

Multiple spaces after comma

#### E242 - tab-after-comma

Tab after comma

#### E251 - unexpected-spaces-around-keyword-parameter-equals

Unexpected spaces around keyword / parameter equals

#### E252 - missing-whitespace-around-parameter-equals

Missing whitespace around parameter equals

#### E261 - too-few-spaces-before-inline-comment

Insert at least two spaces before an inline comment

#### E262 - no-space-after-inline-comment

Inline comment should start with `# `

#### E265 - no-space-after-block-comment

Block comment should start with `# `

#### E266 - multiple-leading-hashes-for-block-comment

Too many leading `#` before block comment

#### E271 - multiple-spaces-after-keyword

Multiple spaces after keyword

#### E272 - multiple-spaces-before-keyword

Multiple spaces before keyword

#### E273 - tab-after-keyword

Tab after keyword

#### E274 - tab-before-keyword

Tab before keyword

#### E275 - missing-whitespace-after-keyword

Missing whitespace after keyword

#### E301 - blank-line-between-methods

Expected {BLANK_LINES_NESTED_LEVEL:?} blank line, found 0

#### E302 - blank-lines-top-level

Expected {expected_blank_lines:?} blank lines, found {actual_blank_lines}

#### E303 - too-many-blank-lines

Too many blank lines ({actual_blank_lines})

#### E304 - blank-line-after-decorator

Blank lines found after function decorator ({lines})

#### E305 - blank-lines-after-function-or-class

Expected 2 blank lines after class or function definition, found ({blank_lines})

#### E306 - blank-lines-before-nested-definition

Expected 1 blank line before a nested definition, found 0

#### E401 - multiple-imports-on-one-line

Multiple imports on one line

#### E402 - module-import-not-at-top-of-file

Module level import not at top of cell

#### E501 - line-too-long

Line too long ({width} > {limit})

#### E502 - redundant-backslash

Redundant backslash

#### E701 - multiple-statements-on-one-line-colon

Multiple statements on one line (colon)

#### E702 - multiple-statements-on-one-line-semicolon

Multiple statements on one line (semicolon)

#### E703 - useless-semicolon

Statement ends with an unnecessary semicolon

#### E711 - none-comparison

Comparison to `None` should be `cond is None`

#### E712 - true-false-comparison

Avoid equality comparisons to `True`; use `{cond}:` for truth checks

#### E713 - not-in-test

Test for membership should be `not in`

#### E714 - not-is-test

Test for object identity should be `is not`

#### E721 - type-comparison

Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks

#### E722 - bare-except

Do not use bare `except`

#### E731 - lambda-assignment

Do not assign a `lambda` expression, use a `def`

#### E741 - ambiguous-variable-name

Ambiguous variable name: `{name}`

#### E742 - ambiguous-class-name

Ambiguous class name: `{name}`

#### E743 - ambiguous-function-name

Ambiguous function name: `{name}`

#### E902 - io-error

{message}

#### E999 - syntax-error

SyntaxError

#### EM101 - raw-string-in-exception

Exception must not use a string literal, assign to variable first

#### EM102 - f-string-in-exception

Exception must not use an f-string literal, assign to variable first

#### EM103 - dot-format-in-exception

Exception must not use a `.format()` string directly, assign to variable first

#### ERA001 - commented-out-code

Found commented-out code

#### EXE001 - shebang-not-executable

Shebang is present but file is not executable

#### EXE002 - shebang-missing-executable-file

The file is executable but no shebang is present

#### EXE003 - shebang-missing-python

Shebang should contain `python`, `pytest`, or `uv run`

#### EXE004 - shebang-leading-whitespace

Avoid whitespace before shebang

#### EXE005 - shebang-not-first-line

Shebang should be at the beginning of the file

</details>

### W: pycodestyle - 警告检查

共 7 条规则

#### W191 - tab-indentation

Indentation contains tabs

#### W291 - trailing-whitespace

Trailing whitespace

#### W292 - missing-newline-at-end-of-file

No newline at end of file

#### W293 - blank-line-with-whitespace

Blank line contains whitespace

#### W391 - too-many-newlines-at-end-of-file

Too many newlines at end of {domain}

#### W505 - doc-line-too-long

Doc line too long ({width} > {limit})

#### W605 - invalid-escape-sequence

Invalid escape sequence: `\{ch}`

### F: pyflakes - 逻辑错误

共 92 条规则

#### F401 - unused-import

`{name}` imported but unused; consider using `importlib.util.find_spec` to test for availability

#### F402 - import-shadowed-by-loop-var

Import `{name}` from {row} shadowed by loop variable

#### F403 - undefined-local-with-import-star

`from {name} import *` used; unable to detect undefined names

#### F404 - late-future-import

`from __future__` imports must occur at the beginning of the file

#### F405 - undefined-local-with-import-star-usage

`{name}` may be undefined, or defined from star imports

#### F406 - undefined-local-with-nested-import-star-usage

`from {name} import *` only allowed at module level

#### F407 - future-feature-not-defined

Future feature `{name}` is not defined

#### F501 - percent-format-invalid-format

`%`-format string has invalid format string: {message}

#### F502 - percent-format-expected-mapping

`%`-format string expected mapping but got sequence

#### F503 - percent-format-expected-sequence

`%`-format string expected sequence but got mapping

#### F504 - percent-format-extra-named-arguments

`%`-format string has unused named argument(s): {message}

#### F505 - percent-format-missing-argument

`%`-format string is missing argument(s) for placeholder(s): {message}

#### F506 - percent-format-mixed-positional-and-named

`%`-format string has mixed positional and named placeholders

#### F507 - percent-format-positional-count-mismatch

`%`-format string has {wanted} placeholder(s) but {got} substitution(s)

#### F508 - percent-format-star-requires-sequence

`%`-format string `*` specifier requires sequence

#### F509 - percent-format-unsupported-format-character

`%`-format string has unsupported format character `{char}`

#### F521 - string-dot-format-invalid-format

`.format` call has invalid format string: {message}

#### F522 - string-dot-format-extra-named-arguments

`.format` call has unused named argument(s): {message}

#### F523 - string-dot-format-extra-positional-arguments

`.format` call has unused arguments at position(s): {message}

#### F524 - string-dot-format-missing-arguments

`.format` call is missing argument(s) for placeholder(s): {message}

<details>
<summary>查看更多 72 条规则</summary>

#### F525 - string-dot-format-mixing-automatic

`.format` string mixes automatic and manual numbering

#### F541 - f-string-missing-placeholders

f-string without any placeholders

#### F601 - multi-value-repeated-key-literal

Dictionary key literal `{name}` repeated

#### F602 - multi-value-repeated-key-variable

Dictionary key `{name}` repeated

#### F621 - expressions-in-star-assignment

Too many expressions in star-unpacking assignment

#### F622 - multiple-starred-expressions

Two starred expressions in assignment

#### F631 - assert-tuple

Assert test is a non-empty tuple, which is always `True`

#### F632 - is-literal

Use `==` to compare constant literals

#### F633 - invalid-print-syntax

Use of `>>` is invalid with `print` function

#### F634 - if-tuple

If test is a tuple, which is always `True`

#### F701 - break-outside-loop

`break` outside loop

#### F702 - continue-outside-loop

`continue` not properly in loop

#### F704 - yield-outside-function

`{keyword}` statement outside of a function

#### F706 - return-outside-function

`return` statement outside of a function/method

#### F707 - default-except-not-last

An `except` block as not the last exception handler

#### F722 - forward-annotation-syntax-error

Syntax error in forward annotation: {parse_error}

#### F811 - redefined-while-unused

Redefinition of unused `{name}` from {row}

#### F821 - undefined-name

Undefined name `{name}`. {tip}

#### F822 - undefined-export

Undefined name `{name}` in `__all__`

#### F823 - undefined-local

Local variable `{name}` referenced before assignment

#### F841 - unused-variable

Local variable `{name}` is assigned to but never used

#### F842 - unused-annotation

Local variable `{name}` is annotated but never used

#### F901 - raise-not-implemented

`raise NotImplemented` should be `raise NotImplementedError`

#### FA100 - future-rewritable-type-annotation

Add `from __future__ import annotations` to simplify `{name}`

#### FA102 - future-required-type-annotation

Missing `from __future__ import annotations`, but uses {reason}

#### FAST001 - fast-api-redundant-response-model

FastAPI route with redundant `response_model` argument

#### FAST002 - fast-api-non-annotated-dependency

FastAPI dependency without `Annotated`

#### FAST003 - fast-api-unused-path-parameter

Parameter `{arg_name}` appears in route path, but not in `{function_name}` signature

#### FBT001 - boolean-type-hint-positional-argument

Boolean-typed positional argument in function definition

#### FBT002 - boolean-default-value-positional-argument

Boolean default positional argument in function definition

#### FBT003 - boolean-positional-value-in-call

Boolean positional value in function call

#### FIX001 - line-contains-fixme

Line contains FIXME, consider resolving the issue

#### FIX002 - line-contains-todo

Line contains TODO, consider resolving the issue

#### FIX003 - line-contains-xxx

Line contains XXX, consider resolving the issue

#### FIX004 - line-contains-hack

Line contains HACK, consider resolving the issue

#### FLY002 - static-join-to-f-string

Consider `{expression}` instead of string join

#### FURB101 - read-whole-file

`Path.open()` followed by `read()` can be replaced by `{filename}.{suggestion}`

#### FURB103 - write-whole-file

`Path.open()` followed by `write()` can be replaced by `{filename}.{suggestion}`

#### FURB105 - print-empty-string

Unnecessary empty string passed to `print`

#### FURB110 - if-exp-instead-of-or-operator

Replace ternary `if` expression with `or` operator

#### FURB113 - repeated-append

Use `{suggestion}` instead of repeatedly calling `{name}.append()`

#### FURB116 - f-string-number-format

Replace `{function_name}` call with `{display}`

#### FURB118 - reimplemented-operator

Use `operator.{operator}` instead of defining a {target}

#### FURB122 - for-loop-writes

Use of `{}.write` in a for loop

#### FURB129 - readlines-in-for

Instead of calling `readlines()`, iterate over file object directly

#### FURB131 - delete-full-slice

Prefer `clear` over deleting a full slice

#### FURB132 - check-and-remove-from-set

Use `{suggestion}` instead of check and `remove`

#### FURB136 - if-expr-min-max

Replace `if` expression with `{min_max}` call

#### FURB140 - reimplemented-starmap

Use `itertools.starmap` instead of the generator

#### FURB142 - for-loop-set-mutations

Use of `set.{}()` in a for loop

#### FURB145 - slice-copy

Prefer `copy` method over slicing

#### FURB148 - unnecessary-enumerate

`enumerate` value is unused, use `for x in range(len(y))` instead

#### FURB152 - math-constant

Replace `{literal}` with `math.{constant}`

#### FURB154 - repeated-global

Use of repeated consecutive `{}`

#### FURB156 - hardcoded-string-charset

Use of hardcoded string charset

#### FURB157 - verbose-decimal-constructor

Verbose expression in `Decimal` constructor

#### FURB161 - bit-count

Use of `bin({existing}).count('1')`

#### FURB162 - fromisoformat-replace-z

Unnecessary timezone replacement with zero offset

#### FURB163 - redundant-log-base

Prefer `math.{log_function}({arg})` over `math.log` with a redundant base

#### FURB164 - unnecessary-from-float

Verbose method `{method_name}` in `{constructor}` construction

#### FURB166 - int-on-sliced-str

Use of `int` with explicit `base={base}` after removing prefix

#### FURB167 - regex-flag-alias

Use of regular expression alias `re.{}`

#### FURB168 - isinstance-type-none

Prefer `is` operator over `isinstance` to check if an object is `None`

#### FURB169 - type-none-comparison

When checking against `None`, use `{}` instead of comparison with `type(None)`

#### FURB171 - single-item-membership-test

Membership test against single-item container

#### FURB177 - implicit-cwd

Prefer `Path.cwd()` over `Path().resolve()` for current-directory lookups

#### FURB180 - meta-class-abc-meta

Use of `metaclass=abc.ABCMeta` to define abstract base class

#### FURB181 - hashlib-digest-hex

Use of hashlib's `.digest().hex()`

#### FURB187 - list-reverse-copy

Use of assignment of `reversed` on list `{name}`

#### FURB188 - slice-to-remove-prefix-or-suffix

Prefer `str.removeprefix()` over conditionally replacing with slice.

#### FURB189 - subclass-builtin

Subclassing `{subclass}` can be error prone, use `collections.{replacement}` instead

#### FURB192 - sorted-min-max

Prefer `min` over `sorted()` to compute the minimum value in a sequence

</details>

---

## cpplint 规则 (C/C++)

**总规则数**: 69 条

**GitHub 项目配置**:
- 启用: 所有规则 (69 条)
- 过滤(忽略): 3 条
  - `whitespace/parens` - 括号空格检查
  - `whitespace/indent_namespace` - 命名空间缩进
  - `runtime/references` - 引用检查
- **实际生效**: 66 条规则

### BUILD

**构建相关规则 - 检查头文件、包含顺序、C++版本特性等**

共 18 条规则:

- `build/class` - 类定义检查
- `build/c++11` - C++11 特性使用检查
- `build/c++14` - C++14 特性使用检查
- `build/c++17` - C++17 特性使用检查
- `build/c++20` - C++20 特性使用检查
- `build/c++tr1` - C++ TR1 特性检查
- `build/deprecated` - 废弃特性检查
- `build/endif_comment` - #endif 注释检查
- `build/explicit_make_pair` - 显式 make_pair 检查
- `build/forward_decl` - 前向声明检查
- `build/header_guard` - 头文件保护符检查
- `build/include` - include 语句检查
- `build/include_alpha` - include 字母顺序检查
- `build/include_order` - include 顺序检查
- `build/include_what_you_use` - include 必要性检查
- `build/namespaces` - 命名空间使用检查
- `build/printf_format` - printf 格式检查
- `build/storage_class` - 存储类检查

### LEGAL

**法律相关规则 - 版权声明检查**

共 1 条规则:

- `legal/copyright` - 版权声明检查

### READABILITY

**可读性规则 - 代码可读性和风格检查**

共 15 条规则:

- `readability/alt_tokens` - 替代标记检查
- `readability/braces` - 大括号使用检查
- `readability/casting` - 类型转换检查
- `readability/check` - CHECK 宏使用检查
- `readability/constructors` - 构造函数检查
- `readability/fn_size` - 函数大小检查
- `readability/inheritance` - 继承检查
- `readability/multiline_comment` - 多行注释检查
- `readability/multiline_string` - 多行字符串检查
- `readability/namespace` - 命名空间检查
- `readability/nolint` - nolint 注释检查
- `readability/nul` - NUL 字符检查
- `readability/strings` - 字符串使用检查
- `readability/todo` - TODO 注释检查
- `readability/utf8` - UTF-8 编码检查

### RUNTIME

**运行时规则 - 运行时行为和安全检查**

共 16 条规则:

- `runtime/arrays` - 数组使用检查
- `runtime/casting` - 运行时类型转换检查
- `runtime/explicit` - explicit 关键字检查
- `runtime/int` - 整数类型检查
- `runtime/init` - 初始化检查
- `runtime/invalid_increment` - 无效自增检查
- `runtime/member_string_references` - 成员字符串引用检查
- `runtime/memset` - memset 使用检查
- `runtime/indentation_namespace` - 命名空间缩进检查
- `runtime/operator` - 运算符重载检查
- `runtime/printf` - printf 使用检查
- `runtime/printf_format` - printf 格式检查
- `runtime/references` - 引用使用检查 (已过滤) ⚠️ (已过滤)
- `runtime/string` - 字符串操作检查
- `runtime/threadsafe_fn` - 线程安全函数检查
- `runtime/vlog` - VLOG 使用检查

### WHITESPACE

**空白符规则 - 空格、缩进、换行等格式检查**

共 19 条规则:

- `whitespace/blank_line` - 空行检查
- `whitespace/braces` - 大括号空格检查
- `whitespace/comma` - 逗号空格检查
- `whitespace/comments` - 注释空格检查
- `whitespace/empty_conditional_body` - 空条件体检查
- `whitespace/empty_if_body` - 空 if 体检查
- `whitespace/empty_loop_body` - 空循环体检查
- `whitespace/end_of_line` - 行尾空格检查
- `whitespace/ending_newline` - 文件末尾换行检查
- `whitespace/forcolon` - for 循环冒号空格检查
- `whitespace/indent` - 缩进检查
- `whitespace/indent_namespace` - 命名空间缩进检查 (已过滤) ⚠️ (已过滤)
- `whitespace/line_length` - 行长度检查 (110字符)
- `whitespace/newline` - 换行检查
- `whitespace/operators` - 运算符空格检查
- `whitespace/parens` - 括号空格检查 (已过滤) ⚠️ (已过滤)
- `whitespace/semicolon` - 分号空格检查
- `whitespace/tab` - Tab 字符检查
- `whitespace/todo` - TODO 空格检查

---

## clang-format 配置 (C/C++)

**配置文件**: `.clang-format`

### 基础配置

```yaml
BasedOnStyle: Google
DerivePointerAlignment: false
ColumnLimit: 110
PointerAlignment: Left
```

### 配置说明

| 配置项 | 值 | 说明 |
|--------|-----|------|
| BasedOnStyle | Google | 基于 Google C++ 风格指南 |
| ColumnLimit | 110 | 行长度限制为 110 字符 |
| PointerAlignment | Left | 指针左对齐 (`int* ptr`) |
| DerivePointerAlignment | false | 不自动推导指针对齐方式 |

### Google 风格包含的规则

Google C++ 风格指南包含以下主要规则:

- **缩进**: 2 个空格
- **大括号**: K&R 风格 (函数除外)
- **命名空间**: 不缩进
- **访问修饰符**: 缩进 1 个空格
- **函数参数**: 对齐或每行一个
- **初始化列表**: 冒号前换行
- **注释**: 使用 // 或 /* */
- **空格**: 运算符周围、逗号后等

---

## Pyright 配置 (Python)

**配置文件**: `pyproject.toml` 中的 `[tool.pyright]` 部分

### 配置内容

```toml
[tool.pyright]
include = ["python/pypto", "tests"]
exclude = ["**/__pycache__", "build", "dist"]
extraPaths = ["python"]
typeCheckingMode = "basic"
pythonVersion = "3.9"
reportMissingTypeStubs = false
reportMissingModuleSource = false
reportCallIssue = false
```

### 配置说明

| 配置项 | 值 | 说明 |
|--------|-----|------|
| typeCheckingMode | basic | 基础类型检查模式 |
| pythonVersion | 3.9 | 目标 Python 版本 |
| include | python/pypto, tests | 检查的目录 |
| exclude | __pycache__, build, dist | 排除的目录 |
| reportMissingTypeStubs | false | 不报告缺失的类型存根 |
| reportMissingModuleSource | false | 不报告缺失的模块源码 |
| reportCallIssue | false | 不报告调用问题 |

### Pyright 检查类型

在 `basic` 模式下，Pyright 检查:

- ✓ 类型注解一致性
- ✓ 函数参数类型匹配
- ✓ 返回值类型匹配
- ✓ 变量类型推断
- ✓ 导入模块存在性
- ✓ 属性访问有效性
- ✓ 基础类型错误

---

## 自定义检查规则

### 1. check-headers (版权头检查)

**脚本**: `tests/lint/check_headers.py`

**检查内容**:
- 检查所有源文件是否包含正确的版权头
- 支持的文件类型:
  - Python: `.py`, `.pyi`
  - C/C++: `.c`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.hxx`
  - CMake: `.cmake`, `CMakeLists.txt`
  - Shell: `.sh`
  - 配置: `.toml`

**期望的版权头格式**:

Python 文件:
```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# ...
```

C/C++ 文件:
```cpp
/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * ...
 */
```

### 2. check-english-only (英文注释检查)

**脚本**: `tests/lint/check_english_only.py`

**检查内容**:
- 检查源文件和文档是否只使用英文
- 检测非英文字符:
  - 中文 (CJK Unified Ideographs)
  - 日文 (Hiragana, Katakana)
  - 韩文 (Hangul)
  - 西里尔文 (Cyrillic)
  - 阿拉伯文 (Arabic)
  - 希伯来文 (Hebrew)
  - 泰文 (Thai)

**支持的文件类型**:
- `.py`, `.pyi`, `.cpp`, `.cc`, `.cxx`, `.c`, `.h`, `.hpp`, `.hxx`, `.md`

**排除目录**:
- `3rdparty/`
- `reference/`

### 3. 通用检查 (pre-commit-hooks)

**来源**: https://github.com/pre-commit/pre-commit-hooks (v4.6.0)

| 检查项 | 说明 |
|--------|------|
| check-added-large-files | 防止提交大文件 (默认 >500KB) |
| check-yaml | 验证 YAML 文件格式正确性 |
| end-of-file-fixer | 确保文件末尾有换行符 |
| trailing-whitespace | 删除行尾空格 |

---

## 规则统计总结

| 工具 | 规则数 | 类型 | 自动修复 |
|------|--------|------|----------|
| Ruff | 295 条 | Python 代码质量 | ✓ 支持 |
| cpplint | 66 条 (69-3) | C++ 代码风格 | ✗ 不支持 |
| clang-format | Google 风格 | C++ 代码格式化 | ✓ 支持 |
| Pyright | basic 模式 | Python 类型检查 | ✗ 不支持 |
| check-headers | 1 条 | 版权头检查 | ✗ 不支持 |
| check-english-only | 1 条 | 英文注释检查 | ✗ 不支持 |
| pre-commit-hooks | 4 条 | 通用检查 | ✓ 部分支持 |

**总计**: 约 368 条规则

---

## 参考链接

- [Ruff 规则文档](https://docs.astral.sh/ruff/rules/)
- [cpplint GitHub](https://github.com/cpplint/cpplint)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [clang-format 文档](https://clang.llvm.org/docs/ClangFormat.html)
- [Pyright 文档](https://microsoft.github.io/pyright/)
- [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks)
