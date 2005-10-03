/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkImageMoreRandomConstIteratorWithIndex_h
#define __itkImageMoreRandomConstIteratorWithIndex_h

#include "itkImageConstIteratorWithIndex.h"

namespace itk
{

/** \class ImageMoreRandomConstIteratorWithIndex
 * 
 * \brief This is a copy of ImageRandomConstIteratorWithIndex, but with a better
 * random number generator.
 *
 */

template<typename TImage>
class ITK_EXPORT ImageMoreRandomConstIteratorWithIndex : public ImageConstIteratorWithIndex<TImage>
{
public:
  /** Standard class typedefs. */
  typedef ImageMoreRandomConstIteratorWithIndex Self;
  typedef ImageConstIteratorWithIndex<TImage>  Superclass;
  
  /** Index typedef support. While this was already typdef'ed in the superclass
   * it needs to be redone here for this subclass to compile properly with gcc.
   * Note that we have to rescope Index back to itk::Index to that is it not
   * confused with ImageIterator::Index. */
  typedef typename TImage::IndexType   IndexType;

  /** Region typedef support. While this was already typdef'ed in the superclass
   * it needs to be redone here for this subclass to compile properly with gcc.
   * Note that we have to rescope Region back to itk::ImageRegion so that is
   * it not confused with ImageIterator::Index. */
  typedef typename TImage::RegionType RegionType;
  
  /** Image typedef support. While this was already typdef'ed in the superclass
   * it needs to be redone here for this subclass to compile properly with gcc.
   * Note that we have to rescope Index back to itk::Index to that is it not
   * confused with ImageIterator::Index. */
  typedef TImage ImageType;

  /** PixelContainer typedef support. Used to refer to the container for
   * the pixel data. While this was already typdef'ed in the superclass
   * it needs to be redone here for this subclass to compile properly with gcc. */
  typedef typename TImage::PixelContainer PixelContainer;
  typedef typename PixelContainer::Pointer PixelContainerPointer;
  
  /** Default constructor. Needed since we provide a cast constructor. */
  ImageMoreRandomConstIteratorWithIndex();
  ~ImageMoreRandomConstIteratorWithIndex() {};
  
  /** Constructor establishes an iterator to walk a particular image and a
   * particular region of that image. */
  ImageMoreRandomConstIteratorWithIndex(const ImageType *ptr, const RegionType& region);

  /** Constructor that can be used to cast from an ImageIterator to an
   * ImageMoreRandomConstIteratorWithIndex. Many routines return an ImageIterator but for a
   * particular task, you may want an ImageMoreRandomConstIteratorWithIndex.  Rather than
   * provide overloaded APIs that return different types of Iterators, itk
   * returns ImageIterators and uses constructors to cast from an
   * ImageIterator to a ImageMoreRandomConstIteratorWithIndex. */
  ImageMoreRandomConstIteratorWithIndex( const ImageConstIteratorWithIndex<TImage> &it)
    { this->ImageConstIteratorWithIndex<TImage>::operator=(it); }

  /** Move an iterator to the beginning of the region. */
  void GoToBegin(void)
  {
    this->RandomJump();
    m_NumberOfSamplesDone = 0L;
  }

  /** Move an iterator to one position past the End of the region. */
  void GoToEnd(void)
  {
    this->RandomJump();
    m_NumberOfSamplesDone = m_NumberOfSamplesRequested;
  }

  /** Is the iterator at the beginning of the region? */
  bool IsAtBegin(void) const
    { return (m_NumberOfSamplesDone == 0L) ; }

  /** Is the iterator at the end of the region? */
  bool IsAtEnd(void) const
    { return (m_NumberOfSamplesDone >= m_NumberOfSamplesRequested);  }
 
  /** Increment (prefix) the selected dimension.
   * No bounds checking is performed. \sa GetIndex \sa operator-- */
  Self & operator++()
  {
    this->RandomJump();
    m_NumberOfSamplesDone++;
    return *this;
  }

  /** Decrement (prefix) the selected dimension.
   * No bounds checking is performed. \sa GetIndex \sa operator++ */
  Self & operator--()
  {
    this->RandomJump();
    m_NumberOfSamplesDone--;
    return *this;
  }
  
  /** Set/Get number of random samples to get from the image region */
  void SetNumberOfSamples( unsigned long number );
  unsigned long GetNumberOfSamples( void ) const;

  /** Reinitialize the seed of the random number generator  */
  static void ReinitializeSeed();
  static void ReinitializeSeed(int);

private:
  void RandomJump();
  unsigned long  m_NumberOfSamplesRequested;
  unsigned long  m_NumberOfSamplesDone;
  unsigned long  m_NumberOfPixelsInRegion;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageMoreRandomConstIteratorWithIndex.txx"
#endif

#endif 



