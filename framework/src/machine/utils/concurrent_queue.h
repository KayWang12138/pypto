/**
 * @file concurrentQueue.h
 * @brief Provides generic support for concurrent queues.
 * @author S. M. Martin
 * @date 26/3/2025
 */

#pragma once

#include <array>
#include "tilefwk/aicpu_common.h"
#include "interface/utils/common.h"
#include "tilefwk/core_func_data.h"

namespace pypto::utils
{

/**
 * @brief Generic class type for concurrent queues
 *
 * Abstracts away the implementation of a concurrent queue, providing thread-safety access.
 * It also attempts to provide low overhead accesses by avoiding the need of mutex mechanisms, in favor of atomics.
 *
 * @tparam T Represents a item type to be stored in the queue
 * @tparam D Represents the NIL value; meant to signify no more elements contained in the queue
 */
template <class T, T D, uint32_t N>
class ConcurrentQueue
{
  public:

  /**
   * Default constructor
   * 
   * \param[in] maxEntries Indicates the maximum amount of entries
  */
  ConcurrentQueue() {}
  ~ConcurrentQueue() {}

  /**
   * Function to push new objects in the queue. This is a thread-safe lock-free operation
   *
   * \param[in] obj The object to push into the queue.
   */
  inline void push(const T obj)
  {
     _elements[_head++ % N] = obj;
  }

  /**
   * Function to pop an object from the queue. Poping removes an object from the front of the queue and returns it to the caller. This is a thread-safe lock-free operation
   *
   * \return The until-now front object of the queue.
   */
  inline T pop()
  {
    return empty() ? D : _elements[_tail++ % N];
  }

  /*
   * Function to determine whether the queue is currently empty or not
   *
   * The past tense in "was" is deliverate, since it can be proven the return value had that
   * value but it cannot be guaranteed that it still has it
   * 
   * \return True, if it is empty; false if its not. Possibly changed now.
   */
  [[nodiscard]] inline bool empty() const { return _head == _tail; }

  /**
   * Function to determine the current possible size of the queue
   *
   * The past tense in "was" is deliverate, since it can be proven the return value had that
   * value but it cannot be guaranteed that it still has it
   * 
   * \return The size of the queue, as last read by this thread. Possibly changed now.
   */
  [[nodiscard]] inline size_t size() const { return _head - _tail; }

  inline void lock()   { while (!__sync_bool_compare_and_swap(&_lock, 0, 1)) {} }
  inline void unlock() { while (!__sync_bool_compare_and_swap(&_lock, 1, 0)) {} }

  private:

  std::array<T, N> _elements;
  size_t _head = 0;
  size_t _tail = 0;
  uint8_t _lock = 0;

};

} // namespace


