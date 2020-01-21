package gr.auth.csd.dws

class Relevance(val search: Search, val product: Product, val relevance: Double) {

  def isRelevant(): Boolean = {
    relevance > 2
  }

}
