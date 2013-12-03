/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxInterpolatorBase_h
#define __elxInterpolatorBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"

#include "itkInterpolateImageFunction.h"


namespace elastix
{

  /**
   * \class InterpolatorBase
   * \brief This class is the elastix base class for all Interpolators.
   *
   * This class contains all the common functionality for Interpolators.
   *
   * \ingroup Interpolators
   * \ingroup ComponentBaseClasses
   */

  template <class TElastix>
    class InterpolatorBase : public BaseComponentSE<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef InterpolatorBase            Self;
    typedef BaseComponentSE<TElastix>   Superclass;

    /** Run-time type information (and related methods). */
    itkTypeMacro( InterpolatorBase, BaseComponentSE );

    /** Typedefs inherited from Elastix. */
    typedef typename Superclass::ElastixType            ElastixType;
    typedef typename Superclass::ElastixPointer         ElastixPointer;
    typedef typename Superclass::ConfigurationType      ConfigurationType;
    typedef typename Superclass::ConfigurationPointer   ConfigurationPointer;
    typedef typename Superclass::RegistrationType       RegistrationType;
    typedef typename Superclass::RegistrationPointer    RegistrationPointer;

    /** Other typedef's. */
    typedef typename ElastixType::MovingImageType       InputImageType;
    typedef typename ElastixType::CoordRepType          CoordRepType;

    /** ITKBaseType. */
    typedef itk::InterpolateImageFunction<
      InputImageType, CoordRepType>                     ITKBaseType;

    /** Cast to ITKBaseType. */
    virtual ITKBaseType * GetAsITKBaseType(void)
    {
      return dynamic_cast<ITKBaseType *>(this);
    }

    /** Cast to ITKBaseType, to use in const functions. */
    virtual const ITKBaseType * GetAsITKBaseType(void) const
    {
      return dynamic_cast<const ITKBaseType *>(this);
    }

  protected:

    /** The constructor. */
    InterpolatorBase() {}
    /** The destructor. */
    virtual ~InterpolatorBase() {}

  private:

    /** The private constructor. */
    InterpolatorBase( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );    // purposely not implemented

  };


} // end namespace elastix



#ifndef ITK_MANUAL_INSTANTIATION
#include "elxInterpolatorBase.hxx"
#endif

#endif // end #ifndef __elxInterpolatorBase_h

