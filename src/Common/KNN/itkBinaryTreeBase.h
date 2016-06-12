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
#ifndef __itkBinaryTreeBase_h
#define __itkBinaryTreeBase_h

#include "itkDataObject.h"

namespace itk
{

/**
 * \class BinaryTreeBase
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template< class TListSample >
class BinaryTreeBase : public DataObject
{
public:

  /** Standard itk. */
  typedef BinaryTreeBase             Self;
  typedef DataObject                 Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** ITK type info. */
  itkTypeMacro( BinaryTreeBase, DataObject );

  /** Typedef's. */
  typedef TListSample SampleType;

  /** Typedef's. */
  typedef typename SampleType::MeasurementVectorType      MeasurementVectorType;
  typedef typename SampleType::MeasurementVectorSizeType  MeasurementVectorSizeType;
  typedef typename SampleType::TotalAbsoluteFrequencyType TotalAbsoluteFrequencyType;

  /** Set and get the samples: the array of points. */
  itkSetObjectMacro( Sample, SampleType );
  itkGetConstObjectMacro( Sample, SampleType );

  /** Get the number of data points. */
  TotalAbsoluteFrequencyType GetNumberOfDataPoints( void ) const;

  /** Get the actual number of data points. */
  TotalAbsoluteFrequencyType GetActualNumberOfDataPoints( void ) const;

  /** Get the dimension of the input data. */
  MeasurementVectorSizeType GetDataDimension( void ) const;

  /** Generate the tree. */
  virtual void GenerateTree( void ) = 0;

protected:

  /** Constructor. */
  BinaryTreeBase();

  /** Destructor. */
  virtual ~BinaryTreeBase() {}

  /** PrintSelf. */
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  BinaryTreeBase( const Self & );   // purposely not implemented
  void operator=( const Self & );   // purposely not implemented

  /** Store the samples. */
  typename SampleType::Pointer m_Sample;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryTreeBase.hxx"
#endif

#endif // end #ifndef __itkBinaryTreeBase_h
