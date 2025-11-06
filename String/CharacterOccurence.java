//4. Find Char Occurrence

import java.util.*;

public class CharacterOccurence{
	public static void main(String... args){
		String input = "zds infotech";

		Map<Character, Integer> map = new HashMap<>();
		char[] chars = input.toCharArray();
		for(char ch : chars){
			if(!map.containsKey(ch)){
				map.put(ch, 1);
			}else {
				int value = map.get(ch);
				map.put(ch, value+1);
			}
		}//end of for loop

		System.out.println(map);
	}
}