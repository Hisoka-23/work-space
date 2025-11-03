import java.util.*;

public class MaxWorldLength{
	public static void main(String[] args){
		String s = "The purple umbrella danced across the street as the wind decided it wanted a partner.";

		String[] str = s.split(" ");

		int count = 0; 

		String maxWordLength = "";

		for(int i=0; i<str.length; i++){
			if(str[i].length() > count){
				count = str[i].length();
				maxWordLength = str[i];
			}
		}
		System.out.println(count + "---" + maxWordLength);
	}
}