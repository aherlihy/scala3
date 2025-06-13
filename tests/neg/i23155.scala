import scala.NamedTuple
//object Test {
//  val explicit: (_1: Int, _1: Int) = ???
//  val viaTupleN: NamedTuple.NamedTuple[Tuple2["_1", "_1"], Tuple2[Int, Int]] = ???
//  val viaTupleExpl: NamedTuple.NamedTuple[("_1", "_1"), (Int, Int)] = ???
//  val viaCons: NamedTuple.NamedTuple["_1" *: "_1" *: EmptyTuple, Int *: Int *: EmptyTuple] = ???
//}

object Unpack {
  (1, 2) match {
    case Unpack(first, _) => first
  }
//  def unapply(e: (Int, Int)): Some[NamedTuple.NamedTuple[("x", "y"), (Int, Int)]] = ??? // error
  def unapply(e: (Int, Int)): Some[NamedTuple.NamedTuple["x" *: "y" *: EmptyTuple, Int *: Int *: EmptyTuple]] = ??? // error
//  def unapply(e: (Int, Int)): Some[(Int, Int)] = ??? // error
//  def unapply(e: (Int, Int)): Some[Int *: Int *: EmptyTuple] = ??? // error
}

//object Unpack {
//  def unapply(e: (Int, Int)): Some[NamedTuple["_1" *: "_2" *: EmptyTuple, Int *: Int *: EmptyTuple]] = ???
//
//  def select[T, R](f: T => R) = ???
//
//  select[(Int, Int), Int] { case Unpack(first, _) => first }
//}

//object Unpack {

  //  def select[T, R](f: T => R) = ???
  //  select[(Int, Int), Int] { case Unpack(first, _) => first }
//  (1, 2) match {
//    case Unpack(first, _) => first
//  }
//  def unapply(e: (Int, Int)): Option[T] = ??? // error

    //  def unapply(e: (Int, Int)): Some[T] = ??? // Same error, includes infinite loop
    //  def unapply(e: (Int, Int)): Some[NamedTuple.NamedTuple["x" *: "y" *: EmptyTuple, Int *: Int *: EmptyTuple]] = ???
    //  def unapply(e: (Int, Int)): Some[NamedTuple.NamedTuple[("_1", "_2"), (Int, Int)]] = ??? // no error
    //  def unapply(e: (Int, Int)): Some[NamedTuple.NamedTuple[Tuple2["_1", "_2"], Tuple2[Int, Int]]] = ??? // no error
    //  def unapply(e: (Int, Int)): Some[(x: Int, y: Int)] = ??? // can't use _1 as key name
    //  def unapply(e: (Int, Int)): Some[NamedTuple.NamedTuple["_1" *: "_2" *: EmptyTuple, Int *: Int *: EmptyTuple]] = ??? // Different crash! Same with not _1
    /*
     * unhandled exception while running MegaPhase{protectedAccessors, extmethods, uncacheGivenAliases, checkStatic, elimByName, hoistSuperArgs, forwardDepChecks, specializeApplyMethods, tryCatchPatterns, patternMatcher} on tests/pos/i23156.scala
     *
     * An unhandled exception was thrown in the compiler.
     * Please file a crash report here:
     * https://github.com/scala/scala3/issues/new/choose
     * For non-enriched exceptions, compile with -Xno-enrich-error-messages.
     *
     * Exception in thread "main" scala.MatchError: List() (of class scala.collection.immutable.Nil$)
     */

    //  def select(f: NamedTuple.NamedTuple[("_1", "_2"), (Int, Int)]): Unit = ??? // fine
    //  def select(f: (_1: Int, _2: Int)): Unit = ??? //    _2 cannot be used as the name of a tuple element because it is a regular tuple selector


    //  def unapply(e: Int *: Int *: EmptyTuple): Some[Int *: Int *: EmptyTuple] = ??? // this fails
    //def unapply(e: (Int, Int)): Some[(Int, Int)] = ??? // this passes
//}