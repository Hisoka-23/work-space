package in.prakash.rest;

import java.time.Duration;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@RestController
public class MyRestController {

	@GetMapping("/single")
	public Mono<String> singleRespone(){
		return Mono.just("Hello World");
	}
	
	@GetMapping("/multiple")
	public Flux<Integer> multipleResponse(){
		return Flux.range(1, 10).delayElements(Duration.ofMinutes(1));
	}
	
}
