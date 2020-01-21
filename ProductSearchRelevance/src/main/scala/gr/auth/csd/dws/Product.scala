package gr.auth.csd.dws

class Product(val uid:String, val title:String, val description:String) {

  def printProductInfo: Unit = {
    print("Product UID: " + uid)
    print("\t//\tProduct Title: " + title)
  }

  def printProductDescrption: Unit = {
    printProductInfo
    println("\t//\tProduct Description: " + description)
  }

}
