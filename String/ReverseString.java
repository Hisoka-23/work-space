//1. Reverse String

import java.util.*;

public class ReverseString{
	public static void main(String... args){
		String str = "hello";

		//Approach - 1
		char[] chArr = str.toCharArray();
		for(int i=chArr.length -1; i>=0; i--){
		System.out.print(chArr[i]);
		}

		System.out.println();

		//Approach - 2
		for(int i=str.length() -1; i>=0; i--){
			System.out.print(str.charAt(i));
		}

		System.out.println();

		//Approach - 3
		StringBuffer sb = new StringBuffer(str);
		System.out.print(sb.reverse());

		System.out.println();

		//Approach - 4
		StringBuilder sbuilder = new StringBuilder(str);
		System.out.print(sbuilder.reverse());

	}
}