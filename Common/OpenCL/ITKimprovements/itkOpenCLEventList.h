/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkOpenCLEventList_h
#define itkOpenCLEventList_h

#include "itkOpenCLEvent.h"
#include <vector>

namespace itk
{
/** \class OpenCLEventList
 * \brief OpenCLEventList class represents a list of OpenCLEvent objects.
 *
 * \ingroup OpenCL
 * \sa OpenCLEvent
 */
class ITKOpenCL_EXPORT OpenCLEventList
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLEventList;

  using OpenCLEventListArrayType = std::vector<cl_event>;

  /** Constructs an empty list of OpenCL events. */
  OpenCLEventList() = default;

  /** Constructs a list of OpenCL events that contains event. If event is null,
   * this constructor will construct an empty list.
   * \sa Append() */
  OpenCLEventList(const OpenCLEvent & event);

  /** Constructs a copy of other.
   * \sa operator=() */
  OpenCLEventList(const OpenCLEventList & other);

  /** Destroys this list of OpenCL events. */
  ~OpenCLEventList();

  /** Assigns the contents of other to this object. */
  OpenCLEventList &
  operator=(const OpenCLEventList & other);

  /** Returns true if this is an empty list, false otherwise.
   * \sa GetSize() */
  bool
  IsEmpty() const
  {
    return this->m_Events.empty();
  }

  /** Returns the size of this event list.
   * \sa IsEmpty(), Get() */
  std::size_t
  GetSize() const
  {
    return this->m_Events.size();
  }

  /** Appends event to this list of OpenCL events if it is not null.
   * Does nothing if event is null.
   * \sa Remove() */
  void
  Append(const OpenCLEvent & event);

  /** Appends the contents of other to this event list. */
  void
  Append(const OpenCLEventList & other);

  /** Removes event from this event list.
   * \sa Append(), Contains() */
  void
  Remove(const OpenCLEvent & event);

  /** Returns the event at index in this event list, or a null OpenCLEvent
   * if index is out of range.
   * \sa GetSize(), Contains() */
  OpenCLEvent
  Get(const std::size_t index) const;

  /** Returns true if this event list contains event, false otherwise.
   * \sa Get(), Remove() */
  bool
  Contains(const OpenCLEvent & event) const;

  /** Returns a const pointer to the raw OpenCL event data in this event list;
   * null if the list is empty. This function is intended for use with native
   * OpenCL library functions that take an array of cl_event objects as
   * an argument.
   * \sa GetSize() */
  const cl_event *
  GetEventData() const;

  /** Returns a const reference to the array of OpenCL events.
   * \sa GetSize() */
  const OpenCLEventListArrayType &
  GetEventArray() const;

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList &
  operator+=(const OpenCLEvent & event);

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList &
  operator+=(const OpenCLEventList & other);

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList &
  operator<<(const OpenCLEvent & event);

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList &
  operator<<(const OpenCLEventList & other);

  /** Waits for all of the events in this list to be signaled as finished.
   * The calling thread is blocked until all of the events are signaled.
   * If the list is empty, then this function returns immediately.
   * \sa OpenCLEvent::WaitForFinished() */
  cl_int
  WaitForFinished();

private:
  OpenCLEventListArrayType m_Events;
};

/** Operator ==
 * Returns true if \a lhs OpenCL event list is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLEventList & lhs, const OpenCLEventList & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL event list is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLEventList & lhs, const OpenCLEventList & rhs);

/** Stream out operator for OpenCLEventList */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLEventList & eventlist)
{
  if (!eventlist.GetSize())
  {
    strm << "OpenCLEventList()";
    return strm;
  }

  OpenCLEventList::OpenCLEventListArrayType::const_iterator it;
  const OpenCLEventList::OpenCLEventListArrayType &         eventsArray = eventlist.GetEventArray();

  std::size_t id = 0;
  strm << "OpenCLEventList contains:" << std::endl;
  for (it = eventsArray.begin(); it < eventsArray.end(); ++it)
  {
    // const OpenCLEvent &event = *it;
    // strm << "array id: " << id << " " << event << std::endl;

    // Let's print only address, printing the OpenCLEvent
    // could halt execution of the program
    strm << "array id: " << id << " " << *it << std::endl;
    ++id;
  }

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLEventList_h */
