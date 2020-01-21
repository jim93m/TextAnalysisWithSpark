package gr.auth.csd.dws

class Search(val id:Int, val term:String, val products: List[Product]) {

  def printSearchInfo{
    print("Search ID: " + id)
    print("\t//\tSearch Term: " + term)
  }

  def printListOfProducts{
    printSearchInfo
    println("--Products--")
    products.foreach(p=> p.printProductInfo)
    println("------------")
  }

}
