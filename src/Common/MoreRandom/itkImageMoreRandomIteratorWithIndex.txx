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
#ifndef _itkImageMoreRandomIteratorWithIndex_txx
#define _itkImageMoreRandomIteratorWithIndex_txx

#include "itkImageMoreRandomIteratorWithIndex.h"

namespace itk
{



template< typename TImage >
ImageMoreRandomIteratorWithIndex<TImage>
::ImageMoreRandomIteratorWithIndex()
  : ImageMoreRandomConstIteratorWithIndex<TImage>() 
{


}



template< typename TImage >
ImageMoreRandomIteratorWithIndex<TImage>
::ImageMoreRandomIteratorWithIndex(ImageType *ptr, const RegionType& region) :
  ImageMoreRandomConstIteratorWithIndex<TImage>(   ptr, region ) 
{


}


 
template< typename TImage >
ImageMoreRandomIteratorWithIndex<TImage>
::ImageMoreRandomIteratorWithIndex( const ImageIteratorWithIndex<TImage> &it):
  ImageMoreRandomConstIteratorWithIndex<TImage>(it)
{ 
}

 
template< typename TImage >
ImageMoreRandomIteratorWithIndex<TImage>
::ImageMoreRandomIteratorWithIndex( const ImageMoreRandomConstIteratorWithIndex<TImage> &it):
  ImageMoreRandomConstIteratorWithIndex<TImage>(it)
{ 
}

 
template< typename TImage >
ImageMoreRandomIteratorWithIndex<TImage> &
ImageMoreRandomIteratorWithIndex<TImage>
::operator=( const ImageMoreRandomConstIteratorWithIndex<TImage> &it)
{ 
  this->ImageMoreRandomConstIteratorWithIndex<TImage>::operator=(it);
  return *this;
}



} // end namespace itk

#endif
