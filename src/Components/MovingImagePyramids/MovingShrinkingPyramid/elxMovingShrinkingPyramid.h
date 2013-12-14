/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxMovingShrinkingPyramid_h
#define __elxMovingShrinkingPyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMultiResolutionShrinkPyramidImageFilter.h"

namespace elastix
{

/**
 * \class MovingShrinkingPyramid
 * \brief A pyramid based on the itk::MultiResolutionShrinkPyramidImageFilter.
 *
 * The parameters used in this class are:
 * \parameter FixedImagePyramid: Select this pyramid as follows:\n
 *    <tt>(MovingImagePyramid "MovingShrinkingImagePyramid")</tt>
 *
 * \ingroup ImagePyramids
 */

template< class TElastix >
class MovingShrinkingPyramid :
  public
  itk::MultiResolutionShrinkPyramidImageFilter<
  typename MovingImagePyramidBase< TElastix >::InputImageType,
  typename MovingImagePyramidBase< TElastix >::OutputImageType >,
  public
  MovingImagePyramidBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef MovingShrinkingPyramid Self;
  typedef itk::MultiResolutionShrinkPyramidImageFilter<
    typename MovingImagePyramidBase< TElastix >::InputImageType,
    typename MovingImagePyramidBase< TElastix >::OutputImageType > Superclass1;
  typedef MovingImagePyramidBase< TElastix > Superclass2;
  typedef itk::SmartPointer< Self >          Pointer;
  typedef itk::SmartPointer< const Self >    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MovingShrinkingPyramid, itk::MultiResolutionShrinkPyramidImageFilter );

  /** Name of this class.
   * Use this name in the parameter file to select this specific pyramid. \n
   * example: <tt>(MovingImagePyramid "MovingShrinkingImagePyramid")</tt>\n
   */
  elxClassNameMacro( "MovingShrinkingImagePyramid" );

  /** Get the ImageDimension. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::InputImageType         InputImageType;
  typedef typename Superclass1::OutputImageType        OutputImageType;
  typedef typename Superclass1::InputImagePointer      InputImagePointer;
  typedef typename Superclass1::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass1::InputImageConstPointer InputImageConstPointer;
  typedef typename Superclass1::ScheduleType           ScheduleType;

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

protected:

  /** The constructor. */
  MovingShrinkingPyramid() {}
  /** The destructor. */
  virtual ~MovingShrinkingPyramid() {}

private:

  /** The private constructor. */
  MovingShrinkingPyramid( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );         // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMovingShrinkingPyramid.hxx"
#endif

#endif // end #ifndef __elxMovingShrinkingPyramid_h
