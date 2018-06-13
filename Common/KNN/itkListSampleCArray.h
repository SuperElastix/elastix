/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkListSampleCArray_h
#define __itkListSampleCArray_h

#include "itkObjectFactory.h"
//#include "itkListSampleBase.h"
#include "itkSample.h"

namespace itk
{
namespace Statistics
{

/**
 * \class ListSampleCArray
 *
 * \brief A ListSampleBase that internally uses a CArray, which can be accessed
 *
 * This class is useful if some function expects a c-array, but you would
 * like to keep things as much as possible in the itk::Statistics-framework.
 *
 * \todo: the second template argument should be removed, since the GetMeasurementVector
 * method is incorrect when TMeasurementVector::ValueType != TInternalValue.
 *
 * \ingroup Miscellaneous
 */

template< class TMeasurementVector, class TInternalValue = typename TMeasurementVector::ValueType >
class ListSampleCArray :
  public Sample< TMeasurementVector >
{
public:

  /** Standard itk. */
  typedef ListSampleCArray             Self;
  typedef Sample< TMeasurementVector > Superclass;
  typedef SmartPointer< Self >         Pointer;
  typedef SmartPointer< const Self >   ConstPointer;

  /** New method for creating an object using a factory.*/
  itkNewMacro( Self );

  /** ITK type info */
  itkTypeMacro( ListSampleCArray, Sample );

  /** Typedef's from Superclass. */
  typedef typename Superclass::MeasurementVectorType      MeasurementVectorType;
  typedef typename Superclass::MeasurementVectorSizeType  MeasurementVectorSizeType;
  typedef typename Superclass::MeasurementType            MeasurementType;
  typedef typename Superclass::AbsoluteFrequencyType      AbsoluteFrequencyType;
  typedef typename Superclass::TotalAbsoluteFrequencyType TotalAbsoluteFrequencyType;
  typedef typename Superclass::InstanceIdentifier         InstanceIdentifier;

  /** Typedef's for the internal data container. */
  typedef TInternalValue      InternalValueType;
  typedef InternalValueType * InternalDataType;
  typedef InternalDataType *  InternalDataContainerType;

  /** Macro to get the internal data container. */
  itkGetConstMacro( InternalContainer, InternalDataContainerType );

  /** Function to resize the data container. */
  void Resize( unsigned long n );

  /** Function to set the actual (not the allocated) size of the data container. */
  void SetActualSize( unsigned long n );

  /** Function to get the actual (not the allocated) size of the data container. */
  unsigned long GetActualSize( void );

  /** Function to clear the data container. */
  void Clear( void );

  /** Function to get the size of the data container. */
  virtual InstanceIdentifier Size( void ) const
  {
    return this->m_InternalContainerSize;
  }


  /** Function to get a point from the data container.
   * NB: the reference to the returned value remains only valid until the next
   * call to this function.
   * The method GetMeasurementVector( const InstanceIdentifier &id, MeasurementVectorType & mv)
   * is actually a preferred way to get a measurement vector.
   */
  virtual const MeasurementVectorType & GetMeasurementVector(
    InstanceIdentifier id ) const;

  /** Function to get a point from the data container. */
  void GetMeasurementVector( InstanceIdentifier id,
    MeasurementVectorType & mv ) const;

  /** Function to set part of a point (measurement) in the data container. */
  void SetMeasurement( InstanceIdentifier id,
    unsigned int dim, const MeasurementType & value );

  /** Function to set a point (measurement vector) in the data container. */
  void SetMeasurementVector( InstanceIdentifier id,
    const MeasurementVectorType & mv );

  /** Function to get the frequency of point i. 1.0 if it exist, 0.0 otherwise. */
  virtual AbsoluteFrequencyType GetFrequency( InstanceIdentifier id ) const;

  /** Function to get the total frequency. */
  virtual TotalAbsoluteFrequencyType GetTotalFrequency( void ) const
  {
    return static_cast< TotalAbsoluteFrequencyType >( this->m_InternalContainerSize );
  }


protected:

  ListSampleCArray();
  virtual ~ListSampleCArray();
  void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  ListSampleCArray( const Self & );   // purposely not implemented
  void operator=( const Self & );     // purposely not implemented

  /** The internal storage of the data in a C array. */
  InternalDataContainerType m_InternalContainer;
  InstanceIdentifier        m_InternalContainerSize;
  InstanceIdentifier        m_ActualSize;

  /** Dummy needed for GetMeasurementVector(). */
  mutable MeasurementVectorType m_TemporaryMeasurementVector;

  /** Function to allocate the memory of the data container. */
  void AllocateInternalContainer( unsigned long size, unsigned int dim );

  /** Function to deallocate the memory of the data container. */
  void DeallocateInternalContainer( void );

};

} // end namespace Statistics
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkListSampleCArray.hxx"
#endif

#endif // end #ifndef __itkListSampleCArray_h
