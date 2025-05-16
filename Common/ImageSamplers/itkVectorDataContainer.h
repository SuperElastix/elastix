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

/** This class is a modification of an ITK class.
 * The original copyright message is pasted here, which includes also
 * the version information: */
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Date:      $Date: 2008-04-15 19:54:41 +0200 (Tue, 15 Apr 2008) $
  Version:   $Revision: 1573 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkVectorDataContainer_h
#define itkVectorDataContainer_h

#include "itkDataObject.h"
#include "itkObjectFactory.h"
#include "itkVectorContainer.h"

#include <utility>
#include <vector>

namespace itk
{

/** \class VectorDataContainer
 *
 * \brief Define a front-end to the STL "vector" container that conforms to the
 * IndexedContainerInterface.
 *
 * This is a full-fleged Object, so there is modification time, debug,
 * and reference count information.
 *
 * Template parameter for VectorDataContainer:
 *
 * TElement =
 *   The element type stored in the container.
 *
 * CHANGES: This class is a modification of the itk::VectorContainer. It now
 * inherits from itk::DataObject instead of itk::Object. This way it is easy
 * to use in an itk::ProcessObject (a filter).
 *
 * \ingroup DataRepresentation
 */
template <typename TElement>
class ITK_TEMPLATE_EXPORT VectorDataContainer
  : public DataObject
  , private std::vector<TElement>
{
public:
  /** Standard class typedefs. */
  using Self = VectorDataContainer;
  using Superclass = DataObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Save the template parameters. */
  using ElementIdentifier = SizeValueType;
  using Element = TElement;

private:
  /** Quick access to the STL vector type that was inherited. */
  using VectorType = std::vector<Element>;
  using size_type = typename VectorType::size_type;
  using VectorIterator = typename VectorType::iterator;
  using VectorConstIterator = typename VectorType::const_iterator;

protected:
  /** Provide pass-through constructors corresponding to all the STL
   * vector constructors.  These are for internal use only since this is also
   * an Object which must be constructed through the "New()" routine. */
  VectorDataContainer()
    : DataObject()
    , VectorType()
  {}
  explicit VectorDataContainer(size_type n)
    : DataObject()
    , VectorType(n)
  {}
  VectorDataContainer(size_type n, const Element & x)
    : DataObject()
    , VectorType(n, x)
  {}
  VectorDataContainer(const Self & r)
    : DataObject()
    , VectorType(r)
  {}
  template <typename InputIterator>
  VectorDataContainer(InputIterator first, InputIterator last)
    : DataObject()
    , VectorType(first, last)
  {}

public:
  /** This type is provided to Adapt this container as an STL container */
  using STLContainerType = VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Standard part of every itk Object. */
  itkOverrideGetNameOfClassMacro(VectorDataContainer);

  /** Cast the container to a STL container type */
  STLContainerType &
  CastToSTLContainer()
  {
    return *this;
  }


  /** Cast the container to a const STL container type */
  const STLContainerType &
  CastToSTLConstContainer() const
  {
    return *this;
  }

  using STLContainerType::begin;
  using STLContainerType::end;
  using STLContainerType::rbegin;
  using STLContainerType::rend;
  using STLContainerType::cbegin;
  using STLContainerType::cend;
  using STLContainerType::crbegin;
  using STLContainerType::crend;

  using STLContainerType::size;
  using STLContainerType::max_size;
  using STLContainerType::resize;
  using STLContainerType::capacity;
  using STLContainerType::empty;
  using STLContainerType::reserve;
  using STLContainerType::shrink_to_fit;

  using STLContainerType::operator[];
  using STLContainerType::at;
  using STLContainerType::front;
  using STLContainerType::back;

  using STLContainerType::assign;
  using STLContainerType::push_back;
  using STLContainerType::pop_back;
  using STLContainerType::insert;
  using STLContainerType::erase;
  using STLContainerType::swap;
  using STLContainerType::clear;

  using STLContainerType::get_allocator;

  using typename STLContainerType::reference;
  using typename STLContainerType::const_reference;
  using typename STLContainerType::iterator;
  using typename STLContainerType::const_iterator;
  // already declared before
  // using STLContainerType::size_type;
  using typename STLContainerType::difference_type;
  using typename STLContainerType::value_type;
  using typename STLContainerType::allocator_type;
  using typename STLContainerType::pointer;
  using typename STLContainerType::const_pointer;
  using typename STLContainerType::reverse_iterator;
  using typename STLContainerType::const_reverse_iterator;

  using Iterator = typename VectorContainer<ElementIdentifier, TElement>::Iterator;
  using ConstIterator = typename VectorContainer<ElementIdentifier, TElement>::ConstIterator;

  /* Declare the public interface routines. */

  /**
   * Get a reference to the element at the given index.
   * It is assumed that the index exists, and it will not automatically
   * be created.
   *
   * It is assumed that the value of the element is modified through the
   * reference.
   */
  Element & ElementAt(ElementIdentifier);

  /**
   * Get a reference to the element at the given index.
   * It is assumed that the index exists, and it will not automatically
   * be created.
   *
   */
  const Element & ElementAt(ElementIdentifier) const;

  /**
   * Get a reference to the element at the given index.
   * If the element location does not exist, it will be created with a
   * default element value.
   *
   * It is assumed that the value of the element is modified through the
   * reference.
   */
  Element & CreateElementAt(ElementIdentifier);

  /**
   * Read the element from the given index.
   * It is assumed that the index exists.
   */
  Element GetElement(ElementIdentifier) const;

  /**
   * Set the element value at the given index.
   * It is assumed that the index exists.
   */
  void SetElement(ElementIdentifier, Element);

  /**
   * Set the element value at the given index.
   * If the element location does not exist, it will be created with a
   * default element value.
   */
  void InsertElement(ElementIdentifier, Element);

  /**
   * Check if the index range of the vector is large enough to allow the
   * given index without expansion.
   */
  bool IndexExists(ElementIdentifier) const;

  /**
   * Check if the given index is in range of the vector.  If it is not, return
   * false.  Otherwise, set the element through the pointer (if it isn't NULL),
   * and return true.
   */
  bool
  GetElementIfIndexExists(ElementIdentifier, Element *) const;

  /**
   * Make sure that the index range of the vector is large enough to allow
   * the given index, expanding it if necessary.  The index will contain
   * the default element regardless of whether expansion occurred.
   */
  void CreateIndex(ElementIdentifier);

  /**
   * Delete the element defined by the index identifier.  In practice, it
   * doesn't make sense to delete a vector index.  Instead, this method just
   * overwrite the index with the default element.
   */
  void DeleteIndex(ElementIdentifier);

  /**
   * Get a begin const iterator for the vector.
   */
  ConstIterator
  Begin() const;

  /**
   * Get an end const iterator for the vector.
   */
  ConstIterator
  End() const;

  /**
   * Get a begin iterator for the vector.
   */
  Iterator
  Begin();

  /**
   * Get an end iterator for the vector.
   */
  Iterator
  End();

  /**
   * Get the number of elements currently stored in the vector.
   */
  SizeValueType
  Size() const;

  /**
   * Tell the container to allocate enough memory to allow at least as many
   * elements as the size given to be stored.  In the generic case of ITK
   * containers this is NOT guaranteed to actually allocate any memory, but it
   * is useful if the implementation of the container allocates contiguous
   * storage. In the particular implementation of this VectorDataContainer the call
   * to this method actually allocates memory for the number of elements
   * defined by ElementIdentifier.
   */
  void Reserve(ElementIdentifier);

  /**
   * Tell the container to try to minimize its memory usage for storage of the
   * current number of elements.  This is NOT guaranteed to decrease memory
   * usage. This method is included here mainly for providing a unified API
   * with other containers in the toolkit.
   */
  void
  Squeeze();

  /**
   * Clear the elements. The final size will be zero.
   */
  void
  Initialize() override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVectorDataContainer.hxx"
#endif

#endif // end itkVectorDataContainer_h
