# Instructions

```{note}
Currently, you need to look directly at the code to understand how to type the various commands. To do: write the syntax as `syntax`.
```

## Supported files for Jupyter book

It is possible to add Jupyter Notebooks (`.ipynb`) or markdown (`.md`) files in the book.

## Main markdown commands

- Headers: add title between #s from one to six of them.

- Emphasis: *Italic* or _Italic_, **Bold** or __Bold__, ***Bold and Italic*** or ___Bold and Italic___

- Lists: As in this code. 
    - Indent code for sublist elements.

- Ordered Lists: Use number followed by periods.
    1. item 1
        1. subitem 1.1
    2. item 2

- Links: [ML git repository](https://github.com/vincenzozimb/MLrepo.git)

- Images:

    ![Images](../images/memino.png)

- Blockquote:
    > This is a blockquote.
    >> This is a nested blockquote.

- Footnote:
    [^mylabel]: My footnote text.

    Now create the footnote by [^mylabel].

- Code:
    - `inline code`
    - Indented block of code:
        ```
        print("Hello World!")
        ```
        or indent (result in non-colored code):
        
            print("Hello World!")

- Horizontal Rules (use either `---`, `***` or `___`):
    
    ---

- Tables:

    | Header 1 | Header 2 |
    |----------|----------|
    | Row 1    | Data     |
    | Row 2    | Data     |

- Line brakes: To create a line break, end a line with two or more spaces and press return.

    First line.  
    Second line.

- Equations:
    - inline equation $E=mc^2$
    - Block equation:    
    
    $$
    \int_{\Omega} d\omega = \int_{\partial \Omega} \omega
    $$

    Make sure to leave a first row empty before writing the block equation for proper rendering of the book.

## Roles and Directives

They are like functions, but written in a markup language. They both serve a similar purpose, but:

> Roles are **inline** constructs that add semantic meaning or style to text within your documents.

> Directives are **block-level** elements that introduce special content or functionality.

It is possible to use any directive / role that is available in Sphinx. Some examples follows.

- Here is the `ref` role:  
Click [Introduction](#../intro) for the first page of the book. Can also be typed as [Introduction](../intro.md).

- The proper "role" is actually the `doc` role:  
Click {doc}`../intro` for the first page of the book.

I enclose all the directives in a `tab-set` directive.

::::{tab-set}

:::{tab-item} note
```{note}
Here is a note
```
:::

:::{tab-item} tip
```{tip}
Here is a tip
```
:::

:::{tab-item} hint
```{hint}
Here is a hint
```
:::

:::{tab-item} seealso
```{seealso} 
Topic to look at
```
:::

:::{tab-item} important
```{important}
This is important
```
:::

:::{tab-item} warning
```{warning} 
This is a warning
```
:::

:::{tab-item} caution
```{caution}
Another warning
```
:::

:::{tab-item} attention
```{attention}
Please, be careful 
```
:::

:::{tab-item} danger
```{danger}
Here is a danger
```
:::

:::{tab-item} error
```{error}
Here is a error
```
:::

:::{tab-item} epigraph
```{epigraph}
This is a two-form.

-- Thomas the tank engine
```
:::

:::{tab-item} glossary
```{glossary}
Term one
  An indented explanation of term 1

A second term
  An indented explanation of term2
```
:::

:::{tab-item} topic
```{topic} Topic title
Topic content.
```
:::

::::

---

- Here is a `admonition` directive, wich needs a specified argument:
```{admonition} (a possible argument)
Here is an admonition
```

- Here is an example about how to change style.

To change the style of the admonition, use
```{admonition} Title
:class: note
Note with personalized title. Also possible to use classes referring to any directive.
```

- Here is an example with the `dropdown` class.

```{tip}
:class: dropdown
Here is a tip
```

- Here is an example with the `toggle` directive.

```{toggle}
Some hidden toggle content!

Hello World!
```

- Here is an example about how to cite a directive:
```{hint}
:name: hint-name
Here is a hint
```

Using the name class it is possible to cite [Reference to my hint](#hint-name)

---

- Here are other directives:

:::{versionadded} 1.2.3
Explanation of the new feature.
:::

:::{versionchanged} 1.2.3
Explanation of the change.
:::

:::{deprecated} 1.2.3
Explanation of the deprecation.
:::

The `:::` can be used instead of `` ``` `` also in the previous directives.

- See also [substitutions](https://jupyterbook.org/en/stable/content/content-blocks.html#substitutions-and-variables-in-markdown) to create more complex objects.

## Citations

It it possible cite references that are stored in a `bibtex` file.  
To do this, use the role: {cite}`JaynesIT`, {cite}`howard2024BayesianRGflowneural`.

Moreover, you can insert a bibliography into your page as shown in [](../intro.md)

## Learn more

This is just a simple starter to get you started.
See and learn a lot more at [jupyterbook.org](https://jupyterbook.org).