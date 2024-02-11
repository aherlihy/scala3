//> using options -source future

// !!! Needs to be compiled currently with -Ycheck:all since that avoids problem
// with illegal opaque types in match types (issue #19434).

import language.experimental.modularity
import language.experimental.namedTuples
import NamedTuple.{NamedTuple, AnyNamedTuple}

/* This is a demonstrator that shows how to map regular for expressions to
 * internal data that can be optimized by a query engine. It needs NamedTuples
 * and type classes but no macros. It's so far very provisional and experimental,
 * intended as a basis for further exploration.
 */

/** The type of expressions in the query language, sub-parts of queries */
trait Expr extends Selectable:

  /** The type of the expression's result.
   *  Note: This needs to be a type member if we want to have complex
   *  constructors that introduce dependencies on value parameters in Result.
   *  An example of such a constructor is the commented out version of Select below.
   */
  type Result // result of the subexpression, which could be another expression

  /** This type is used to support selection with any of the field names
   *  defined by Fields.
   */
  type Fields = NamedTuple.Map[NamedTuple.From[Result], Expr.Of]
  /*
    [1,2].map(f) => NamedTuple.Map([1,2], f)
    NamedTuple.From[Result] => creating a named tuple from Result
    f = Expr.Of, => Expr.Of(t where Tuple.From(Result))
    ex:
      case class Person(age: Int, name: String)
      NamedTuple.From[Person] => (age: Int, name: String)
      NamedTuple.Map[^, Expr.Of] => (age: Expr.Of[Int], name: Expr.Of[String])

      case class Foo(person: Person, year: Int)
      val f: Expr.Of[Foo] = ...
      f.person // Expr.Of[Person]
  */

  /** A selection of a field name defined by Fields is implemented by `selectDynamic`.
   *  The implementation will add a cast to the right Expr type corresponding
   *  to the field type.
   */
  def selectDynamic(fieldName: String) = Expr.Select(this, fieldName)

  /** Member methods to implement universal equality on Expr level. */
  def == (other: Expr): Expr.Of[Boolean] = Expr.Eq(this, other)
  def != (other: Expr): Expr.Of[Boolean] = Expr.Ne(this, other)

object Expr:

  /** Convenience alias to regain something close to parameterized Expr types */
  type Of[A] = Expr { type Result = A } // why avoid Expr[T]?

  /** Sample extension methods for individual types */
  extension (x: Expr.Of[Int]) // define methods on the values of Expr
    def > (y: Expr.Of[Int]): Expr.Of[Boolean] = Gt(x, y)
    def > (y: Int): Expr.Of[Boolean] = Gt(x, IntLit(y))
  extension (x: Expr.Of[Boolean])
    def &&(y: Expr.Of[Boolean]): Expr.Of[Boolean] = And(x, y)
    def || (y: Expr.Of[Boolean]): Expr.Of[Boolean] = Or(x, y)
  // would need to add +/- etc.


  // Note: All field names of constructors in the query language are prefixed with `$`
  // so that we don't accidentally pick a field name of a constructor class where we want
  // a name in the domain model instead. => all of these classes are going to have 'domain-level' attributes e.g. "name/address/age",
  // and want to differentiate attributes of the query class vs. attributes of the data

  // Some sample constructors for query operations, e.g. Exprs, i.e. AST of what goes inside the filter function
  case class Gt($x: Expr.Of[Int], $y: Expr.Of[Int]) extends Expr.Of[Boolean]
  case class Plus($x: Expr.Of[Int], $y: Expr.Of[Int]) extends Expr.Of[Int]
  case class And($x: Expr.Of[Boolean], $y: Expr.Of[Boolean]) extends Expr.Of[Boolean]
  case class Or($x: Expr.Of[Boolean], $y: Expr.Of[Boolean]) extends Expr.Of[Boolean]

  // So far Select is weakly typed, so `selectDynamic` is easy to implement.
  // Todo: Make it strongly typed like the other cases, along the lines
  // of the commented out version below. Select like DB, not selectDyamic
  case class Select[A]($x: Expr.Of[A], $name: String) extends Expr // select $name from $x, weakly typed because Expr not Expr.Of

  case class Single[S <: String, A]($x: Expr.Of[A]) //Single[a singleton-key-name, value-type], and pass a subquery that returns an A,
    extends Expr.Of[NamedTuple[S *: EmptyTuple, A *: EmptyTuple]] // *: is like :: for lists, so S*: EmptyTuple is a tuple of length 1 with just S bc there is no syntax for a 1-element tuple
      /* extends: Expr.Of[NamedTuple[Tuple(S), Tuple(A)]]
                   Expr.Of: `Expr { type Result = A }`, gives Expr { type Result = NamedTuple[S, A] }
                    within Expr, `type Fields = NamedTuple.Map[NamedTuple.From[Result], Expr.Of]`
                      assigning fields basically spread on Result, so:
                        NamedTuple.Map[NamedTuple.From[ NamedTuple[S,A] ], Expr.Of ] // sub Result
                        NamedTuple.Map[ NamedTuple[S,A], Expr.Of ]                   // reduce NamedTuple.From
                        NamedTuple[ S, Expr.Of[A] ]                                  // reduce NamedTuple.Map
                        so the computed type of Fields is NamedTuple[ S, Expr.Of[A] ], so Single is a type that has a field S of type Expr.Of[A]

            => wraps a symbolic value $x and gives it a name that can be selected
       */


  case class Concat[A <: AnyNamedTuple, B <: AnyNamedTuple]($x: Expr.Of[A], $y: Expr.Of[B])
    extends Expr.Of[NamedTuple.Concat[A, B]]

  case class Join[A <: AnyNamedTuple](a: A)
    extends Expr.Of[NamedTuple.Map[A, StripExpr]]

  type StripExpr[E] = E match
    case Expr.Of[b] => b

  // Also weakly typed in the arguents since these two classes model universal equality */
  case class Eq($x: Expr, $y: Expr) extends Expr.Of[Boolean]
  case class Ne($x: Expr, $y: Expr) extends Expr.Of[Boolean]

  /** References are placeholders for parameters */
  private var refCount = 0

  case class Ref[A]($name: String = "") extends Expr.Of[A]: // a fresh variable of type A
    val id = refCount
    refCount += 1
    override def toString = s"ref$id(${$name})"

  /** Literals are type-specific, tailored to the types that the DB supports */
  case class IntLit($value: Int) extends Expr.Of[Int]

  /** Scala values can be lifted into literals by conversions */
  given Conversion[Int, IntLit] = IntLit(_)

  /** The internal representation of a function `A => B`
   *  Query languages are usually first-order, so Fun is not an Expr
   *  eg Not higher order so no first class functions so cannot pass Fun where Expr is expected
   */
  case class Fun[A, B](param: Ref[A], f: B)

  type Pred[A] = Fun[A, Expr.Of[Boolean]]

  /** Explicit conversion from
   *      (name_1: Expr.Of[T_1], ..., name_n: Expr.Of[T_n])
   *  to
   *      Expr.Of[(name_1: T_1, ..., name_n: T_n)]
   */
  // called lifting bc value (many exprs) => expr
  // creating a (name: Expr, name2: Expr2) is constructing a query of cartesian project of Expr and Expr2
  // should not be called toRow, more like combineTuples
  // Q: how would you add a join condition here?
  extension [A <: AnyNamedTuple](x: A) def toRow: Join[A] = Join(x)
  //

  /** Same as _.toRow, as an implicit conversion */
  given [A <: AnyNamedTuple] => Conversion[A, Expr.Join[A]] = Expr.Join(_)

end Expr

/** The type of database queries. So far, we have queries
 *  that represent whole DB tables and queries that reify
 *  for-expressions as data.
 */
trait Query[A]

object Query:
  import Expr.{Pred, Fun, Ref}

  case class Filter[A]($q: Query[A], $p: Pred[A]) extends Query[A]
  case class Map[A, B]($q: Query[A], $f: Fun[A, Expr.Of[B]]) extends Query[B]
  case class FlatMap[A, B]($q: Query[A], $f: Fun[A, Query[B]]) extends Query[B]

  // Extension methods to support for-expression syntax for queries (does it have to be an extension?)
  extension [R](x: Query[R]) // x is the query upon which we are calling Map

    def withFilter(p: Ref[R] => Expr.Of[Boolean]): Query[R] =
      val ref = Ref[R]()
      Filter(x, Fun(ref, p(ref)))

    def map[B](f: Ref[R] => Expr.Of[B]): Query[B] = // map takes a function from ref, eg param, to expression of type B, and returns a Query[B]
      val ref = Ref[R]()
      Map(x, Fun(ref, f(ref))) // construct an AST, essentially x.Map(Fun(...))

    def flatMap[B](f: Ref[R] => Query[B]): Query[B] =
      val ref = Ref[R]()
      FlatMap(x, Fun(ref, f(ref)))
end Query

/** The type of query references to database tables */
case class Table[R]($name: String) extends Query[R]

// Everything below is code using the model -----------------------------

// Some sample types
case class City(zipCode: Int, name: String, population: Int)
type Address = (city: City, street: String, number: Int)
type Person = (name: String, age: Int, addr: Address)

@main def Test =

  val cities = Table[City]("cities") // Table of type City, where Table extends Query[City], Table needs $name ("cities") so that the generated SQL can specify which table

  val q1 = cities.map(c =>
    c.zipCode
  )
  // Map(cities, Fun(x: Expr.Of[City], x.zipCode: Expr.Of[Int]): Query[Int]
  // select zipCode from cities
  // How does Select relate to Map? Why is Select a Expr but Map is not?
  def anyf(cityZ: Expr.Of[Int]) = 100
  def sysC(i: Int): Int

  val q2 = cities.withFilter(city => // takes an expr that returns a boolean
      // this next 2 lines is just AST generation and can only do what the DB supports
      val x = city.population + 1000 - 4 // these ops need to be defined on the AST and take Expr type
      city.population > x // 10_000 (_ is the same, just readability)
      // if i wanted to wrap a computation that could not be expressed with Expr (e.g. an external library), then I would need to package the call in a AST node "UDF"
      // UDF(f: a => b, Expr.Of[Fun[A,B]]) => this would be the method that converts general purpose programs into ASTs, e.g. do pushdown, takes a function and returns somethign the DB can interweave with the rest of the query
      // Project(f), f has some capability based compile-time restriction
      city.population > sysC(x)
    ).map(city =>
      city.name
    )

  // original AST: Select(Project(Join)) => Select(f, Project(f2, Join(f3))) where f is a UDF sent to the DB that "behaves" like Select/Project/Join
  //

//  qZ = cities.map(f => ReverseA).map(f => ReverseB) => query optimizer can merge and elide

  val q3 = // q3 gets desugared into q2
    for
      city <- cities
      if city.population > 10_000
    yield city.name

  val q4 = // gets desugared into flatMap, explainTyper. Essentially a self join. How does flatMap get converted into join?
    for
      city <- cities
      alt <- cities
      if city.name == alt.name && city.zipCode != alt.zipCode
    yield
      city

  val addresses = Table[Address]("addresses")
  val q5 =
    for
      city <- cities
      addr <- addresses
      if addr.street == city.name
    yield
      (name = city.name, num = addr.number) // toRow, e.g. a join?

  val q6 =
    cities.map: city =>
      (name = city.name, zipCode = city.zipCode)

  def run[T](q: Query[T]): Iterator[T] = ???

  def x1: Iterator[Int] = run(q1)
  def x2: Iterator[String] = run(q2)
  def x3: Iterator[String] = run(q3)
  def x4: Iterator[City] = run(q4)
  def x5: Iterator[(name: String, num: Int)] = run(q5)
  def x6: Iterator[(name: String, zipCode: Int)] = run(q6)

  // explicit join = matchtype?

  println(q1)
  println(q2)
  println(q3)
  println(q4)
  println(q5)
  println(q6)

/* The following is not needed currently

/** A type class for types that can map to a database table */
trait Row:
  type Self
  type Fields = NamedTuple.From[Self]
  type FieldExprs = NamedTuple.Map[Fields, Expr.Of]

  //def toFields(x: Self): Fields = ???
  //def fromFields(x: Fields): Self = ???

*/