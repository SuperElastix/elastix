/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxMovingRecursivePyramid_h
#define __elxMovingRecursivePyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"

namespace elastix
{

  /**
   * \class MovingRecursivePyramid
   * \brief A pyramid based on the itkRecursiveMultiResolutionPyramidImageFilter.
   *
   * The parameters used in this class are:
   * \parameter MovingImagePyramid: Select this pyramid as follows:\n
   *    <tt>(MovingImagePyramid "MovingRecursiveImagePyramid")</tt>
   *
   * \ingroup ImagePyramids
   */

  template <class TElastix>
    class MovingRecursivePyramid :
    public
      itk::RecursiveMultiResolutionPyramidImageFilter<
        typename MovingImagePyramidBase<TElastix>::InputImageType,
        typename MovingImagePyramidBase<TElastix>::OutputImageType >,
    public
      MovingImagePyramidBase<TElastix>
  {
  public:

    /** Standard ITK. */
    typedef MovingRecursivePyramid                                    Self;
    typedef itk::RecursiveMultiResolutionPyramidImageFilter<
        typename MovingImagePyramidBase<TElastix>::InputImageType,
        typename MovingImagePyramidBase<TElastix>::OutputImageType >  Superclass1;
    typedef MovingImagePyramidBase<TElastix>                          Superclass2;
    typedef itk::SmartPointer<Self>                                   Pointer;
    typedef itk::SmartPointer<const Self>                             ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( MovingRecursivePyramid, RecursiveMultiResolutionPyramidImageFilter );

    /** Name of this class.
     * Use this name in the parameter file to select this specific pyramid. \n
     * example: <tt>(MovingImagePyramid "MovingRecursiveImagePyramid")</tt>\n
     */
    elxClassNameMacro( "MovingRecursiveImagePyramid" );

    /** Get the ImageDimension. */
    itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

    /** Typedefs inherited from Superclass1. */
    typedef typename Superclass1::InputImageType            InputImageType;
    typedef typename Superclass1::OutputImageType           OutputImageType;
    typedef typename Superclass1::InputImagePointer         InputImagePointer;
    typedef typename Superclass1::OutputImagePointer        OutputImagePointer;
    typedef typename Superclass1::InputImageConstPointer    InputImageConstPointer;

    /** Typedefs inherited from Elastix. */
    typedef typename Superclass2::ElastixType           ElastixType;
    typedef typename Superclass2::ElastixPointer        ElastixPointer;
    typedef typename Superclass2::ConfigurationType     ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer  ConfigurationPointer;
    typedef typename Superclass2::RegistrationType      RegistrationType;
    typedef typename Superclass2::RegistrationPointer   RegistrationPointer;
    typedef typename Superclass2::ITKBaseType           ITKBaseType;

  protected:

    /** The constructor. */
    MovingRecursivePyramid() {}
    /** The destructor. */
    virtual ~MovingRecursivePyramid() {}

  private:

    /** The private constructor. */
    MovingRecursivePyramid( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );          // purposely not implemented

  }; // end class MovingRecursivePyramid


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMovingRecursivePyramid.hxx"
#endif

#endif // end #ifndef __elxMovingRecursivePyramid_h
