type Ctx = (Int, Int)

validator main {
  else((i1, i2): Ctx) {
    i1 + i2 == 42
  }
}
