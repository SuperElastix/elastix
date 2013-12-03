/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxOptimizerBase_h
#define __elxOptimizerBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkOptimizer.h"


namespace elastix
{

/**
 * \class OptimizerBase
 * \brief This class is the elastix base class for all Optimizers.
 *
 * This class contains all the common functionality for Optimizers.
 *
 * The parameters used in this class are:
 * \parameter NewSamplesEveryIteration: if this flag is set to "true" some
 *    optimizers ask the metric to select a new set of spatial samples in
 *    every iteration. This, if used in combination with the correct optimizer (such as the
 *    StandardGradientDescent), and ImageSampler (Random, RandomCoordinate, or RandomSparseMask),
 *    allows for a very low number of spatial samples (around 2000), even with large images
 *    and transforms with a large number of parameters.\n
 *    Choose one from {"true", "false"} for every resolution.\n
 *    example: <tt>(NewSamplesEveryIteration "true" "true" "true")</tt> \n
 *    Default is "false" for every resolution.\n
 *
 * \ingroup Optimizers
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class OptimizerBase : public BaseComponentSE<TElastix>
{
public:

  /** Standard ITK-stuff. */
  typedef OptimizerBase               Self;
  typedef BaseComponentSE<TElastix>   Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro( OptimizerBase, BaseComponentSE );

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass::ElastixType            ElastixType;
  typedef typename Superclass::ElastixPointer         ElastixPointer;
  typedef typename Superclass::ConfigurationType      ConfigurationType;
  typedef typename Superclass::ConfigurationPointer   ConfigurationPointer;
  typedef typename Superclass::RegistrationType       RegistrationType;
  typedef typename Superclass::RegistrationPointer    RegistrationPointer;

  /** ITKBaseType. */
  typedef itk::Optimizer  ITKBaseType;

  /** Typedef needed for the SetCurrentPositionPublic function. */
  typedef typename ITKBaseType::ParametersType        ParametersType;

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

  /** Add empty SetCurrentPositionPublic, so this function is known in every inherited class. */
  virtual void SetCurrentPositionPublic( const ParametersType &param );

  /** Execute stuff before each new pyramid resolution:
   * \li Find out if new samples are used every new iteration in this resolution.
   */
  virtual void BeforeEachResolutionBase();

  /** Execute stuff after registration:
   * \li Compute and print MD5 hash of the transform parameters.
   */
  virtual void AfterRegistrationBase( void );

  /** Method that sets the scales defined by a sinus
   * scale[i] = amplitude^( sin(i/nrofparam*2pi*frequency) )
   */
  virtual void SetSinusScales(double amplitude, double frequency,
    unsigned long numberOfParameters);

protected:

  /** The constructor. */
  OptimizerBase();
  /** The destructor. */
  virtual ~OptimizerBase() {}

  /** Force the metric to base its computation on a new subset of image samples.
   * Not every metric may have implemented this.
   */
  virtual void SelectNewSamples(void);

  /** Check whether the user asked to select new samples every iteration. */
  virtual bool GetNewSamplesEveryIteration( void ) const;

private:

  /** The private constructor. */
  OptimizerBase( const Self& );   // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self& );  // purposely not implemented

  /** Member variable to store the user preference for using new
   * samples each iteration.
   */
  bool m_NewSamplesEveryIteration;

};


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxOptimizerBase.hxx"
#endif

#endif // end #ifndef __elxOptimizerBase_h
