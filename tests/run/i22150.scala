//> using options -experimental -language:experimental.namedTuples
import language.experimental.namedTuples

val directionsNT = IArray(
  (dx = 0, dy = 1), // up
  (dx = 1, dy = 0), // right
  (dx = 0, dy = -1), // down
  (dx = -1, dy = 0), // left
)
val IArray(UpNT @ _, _, _, _) = directionsNT
@main def Test =
  println(UpNT.dx)
