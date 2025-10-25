package in.prakash.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import in.prakash.entity.Customer;

public interface CustomerRepository extends JpaRepository<Customer, Integer> {

}
