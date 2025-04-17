(program (1 0 0)
  (lam n
    (force
      (app
        (lam self
          (delay
            (lam n
              (force
                (app
                  (var self)
                  (var self)
                  (var n)
                )
              )
            )
          )
        )
        (lam self
          (delay
            (lam n
              (if
                (app (builtin lessThanEqualsInteger) (var n) (con integer 1))
                (con integer 1)
                (app
                  (builtin multiplyInteger)
                  (var n)
                  (force
                    (app
                      (var self)
                      (var self)
                      (app (builtin subtractInteger) (var n) (con integer 1))
                    )
                  )
                )
              )
            )
          )
        )
        (con integer 5)
      )
    )
  )
) 