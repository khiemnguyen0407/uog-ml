--[[
build.lua: step-by-step "builds" (one figure replacing the previous one) for reveal.js AND Beamer.

Usage in a .qmd file:

    ::: build
    ![](step1.png){width="70%"}

    ![](step2.png){width="70%"}
    :::

Each top-level block inside the div is one step. If a step consists of several blocks (for example a
figure plus an equation), wrap that step in its own div:

    ::: build
    ::: {}
    ![](a.png)
    $$ J = 0 $$
    :::
    ::: {}
    ![](b.png)
    $$ J = 1 $$
    :::
    :::

Output:
  * revealjs: steps are stacked (.r-stack); steps 2..n are fragments. The CSS rule in
    styles/lecture.scss hides a step as soon as a later step is visible, so steps never shine through.
  * beamer:   \begin{overprint} \onslide<+> step 1 ... \onslide<+-> step n \end{overprint}.
    The pause counter is then corrected so that a following `. . .` does not add an empty overlay.
  * any other format (html, docx, pdf, ...): only the final step is shown.

Builds may be combined freely with `. . .` pauses before or after them on the same slide.

The filter also handles `::: small` (smaller text) so that it works in Beamer as well as reveal.js.
]]

local function steps_of(div)
  local steps = pandoc.List()
  for _, blk in ipairs(div.content) do
    steps:insert(blk)
  end
  return steps
end

local function small_div(el)
  -- `::: small` -> smaller text. reveal.js uses the .small CSS class; Beamer needs \footnotesize.
  if quarto.doc.is_format("beamer") then
    local out = pandoc.List({ pandoc.RawBlock("latex", "{\\footnotesize") })
    out:extend(el.content)
    out:insert(pandoc.RawBlock("latex", "\\par}"))
    return out
  end
  return nil
end

function Div(el)
  if el.classes:includes("small") then
    return small_div(el)
  end
  if not el.classes:includes("build") then
    return nil
  end
  local steps = steps_of(el)
  if #steps == 0 then
    return pandoc.Null()
  end

  if quarto.doc.is_format("revealjs") then
    local out = pandoc.List()
    for i, blk in ipairs(steps) do
      local classes = { "build-step" }
      if i > 1 then
        table.insert(classes, "fragment")
      end
      out:insert(pandoc.Div({ blk }, pandoc.Attr("", classes)))
    end
    local attr = pandoc.Attr(el.identifier, { "r-stack", "build-stack" }, el.attributes)
    return pandoc.Div(out, attr)

  elseif quarto.doc.is_format("beamer") then
    local out = pandoc.List({ pandoc.RawBlock("latex", "\\begin{overprint}") })
    for i, blk in ipairs(steps) do
      -- handout mode (Beamer class option "handout"): keep only the final step
      local spec = (i == #steps) and "<+-| handout:1>" or "<+| handout:0>"
      out:insert(pandoc.RawBlock("latex", "\\onslide" .. spec))
      out:insert(blk)
    end
    out:insert(pandoc.RawBlock("latex", "\\end{overprint}\\addtocounter{beamerpauses}{-1}"))
    return out

  else
    return steps[#steps]
  end
end
