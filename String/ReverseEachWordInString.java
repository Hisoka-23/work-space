//3. Reverse Each Word in String

import java.util.*;

public class ReverseEachWordInString{
	public static void main(String... args){
		String input = "java code";
		System.out.println("Original String :: "+input);

		String ouput = "";
		
		String[] words = input.split(" ");

		for(String word: words){

			String revWord = "";
			//reverse word
			for(int i=word.length()-1; i>=0; i--){
				revWord += word.charAt(i);
			}
			ouput += revWord + " ";
		}

		System.out.println("rev String :: "+ouput);
	}
}