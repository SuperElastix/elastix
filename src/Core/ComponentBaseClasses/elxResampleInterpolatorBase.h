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

#ifndef __elxResampleInterpolatorBase_h
#define __elxResampleInterpolatorBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class ResampleInterpolatorBase
 * \brief This class is the elastix base class for all ResampleInterpolators.
 *
 * This class contains all the common functionality for ResampleInterpolators.
 *
 * \ingroup ResampleInterpolators
 * \ingroup ComponentBaseClasses
 */

template< class TElastix >
class ResampleInterpolatorBase : public BaseComponentSE< TElastix >
{
public:

  /** Standard ITK stuff. */
  typedef ResampleInterpolatorBase    Self;
  typedef BaseComponentSE< TElastix > Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro( ResampleInterpolatorBase, BaseComponentSE );

  /** Typedef's from superclass. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Typedef's from elastix. */
  typedef typename ElastixType::MovingImageType InputImageType;
  typedef typename ElastixType::CoordRepType    CoordRepType;

  /** Other typedef's. */
  typedef itk::InterpolateImageFunction<
    InputImageType, CoordRepType >                  ITKBaseType;

  /** Typedef that is used in the elastix dll version. */
  typedef typename ElastixType::ParameterMapType ParameterMapType;

  /** Cast ti ITKBaseType. */
  virtual ITKBaseType * GetAsITKBaseType( void )
  {
    return dynamic_cast< ITKBaseType * >( this );
  }


  /** Cast to ITKBaseType, to use in const functions. */
  virtual const ITKBaseType * GetAsITKBaseType( void ) const
  {
    return dynamic_cast< const ITKBaseType * >( this );
  }


  /** Execute stuff before the actual transformation:
   * \li nothing here
   */
  virtual int BeforeAllTransformix( void ){ return 0; }

  /** Function to read transform-parameters from a file. */
  virtual void ReadFromFile( void );

  /** Function to write transform-parameters to a file. */
  virtual void WriteToFile( void ) const;

  /** Function to create transform-parameters map. */
  virtual void CreateTransformParametersMap( ParameterMapType * paramsMap ) const;

protected:

  /** The constructor. */
  ResampleInterpolatorBase() {}
  /** The destructor. */
  virtual ~ResampleInterpolatorBase() {}

private:

  /** The private constructor. */
  ResampleInterpolatorBase( const Self & );   // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );             // purposely not implemented

};

} //end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxResampleInterpolatorBase.hxx"
#endif

#endif // end #ifndef __elxResampleInterpolatorBase_h
