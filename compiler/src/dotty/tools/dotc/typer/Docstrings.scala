package dotty.tools
package dotc
package typer

import core._
import Contexts._, Symbols._, Decorators._, Comments._
import ast.tpd

trait Docstrings { self: Typer =>

  /**
   * Expands or cooks the documentation for all members of `cdef`.
   *
   * @see Docstrings#cookComment
   */
  def cookComments(cdef: tpd.Tree, cls: ClassSymbol)(implicit ctx: Context): Unit = {
    // val cls = cdef.symbol
    val cookingCtx = ctx.localContext(cdef, cls).setNewScope
    cls.info.allMembers.foreach { member =>
      cookComment(member.symbol, cls)(cookingCtx)
    }
    cookComment(cls, cls)(cookingCtx)
  }

  /**
   * Expands or cooks the documentation for `sym` in class `owner`.
   * The expanded comment will directly replace the original comment in the doc context.
   *
   * The expansion registers `@define` sections, and will replace `@inheritdoc` and variable
   * occurrences in the comments.
   *
   * If the doc comments contain `@usecase` sections, they will be typed.
   *
   * @param sym   The symbol for which the comment is being cooked.
   * @param owner The class for which comments are being cooked.
   */
  def cookComment(sym: Symbol, owner: Symbol)(implicit ctx: Context): Option[Comment] = {
    ctx.docCtx.flatMap { docCtx =>
      expand(sym, owner)(ctx, docCtx)
    }
  }

  private def expand(sym: Symbol, owner: Symbol)(implicit ctx: Context, docCtx: ContextDocstrings): Option[Comment] = {
    docCtx.docstring(sym).flatMap {
      case cmt if cmt.isExpanded =>
        Some(cmt)
      case _ =>
        expandComment(sym).map { expanded =>
          val typedUsecases = expanded.usecases.map { usecase =>
            enterSymbol(createSymbol(usecase.untpdCode))
            typedStats(usecase.untpdCode :: Nil, owner) match {
              case List(df: tpd.DefDef) =>
                usecase.typed(df)
              case _ =>
                ctx.error("`@usecase` was not a valid definition", usecase.codePos)
                usecase
            }
          }

          val commentWithUsecases = expanded.copy(usecases = typedUsecases)
          docCtx.addDocstring(sym, Some(commentWithUsecases))
          commentWithUsecases
        }
    }
  }

  private def expandComment(sym: Symbol, owner: Symbol, comment: Comment)(implicit ctx: Context, docCtx: ContextDocstrings): Comment = {
    val tplExp = docCtx.templateExpander
    tplExp.defineVariables(sym)
    val newComment = comment.expand(tplExp.expandedDocComment(sym, owner, _))
    docCtx.addDocstring(sym, Some(newComment))
    newComment
  }

  private def expandComment(sym: Symbol)(implicit ctx: Context, docCtx: ContextDocstrings): Option[Comment] = {
    if (sym eq NoSymbol) None
    else {
      for {
        cmt <- docCtx.docstring(sym) if !cmt.isExpanded
        _ = expandComment(sym.owner)
      } yield expandComment(sym, sym.owner, cmt)
    }
  }
}
