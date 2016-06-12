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

#ifndef __itkTranslationStackTransformInitializer_h
#define __itkTranslationStackTransformInitializer_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include <iostream>

namespace itk
{

/**
 * \class TranslationStackTransformInitializer
 *
 * \brief TranslationStackTransformInitializer is a helper class intended to
 * initialize the translation of a TranslationTransforms
 *
 * This class is connected to the fixed image, moving image and transform
 * involved in the registration. Two modes of operation are possible:
 *
 * - Geometrical,
 * - Center of mass
 *
 * In the first mode, the geometrical centers of the images are computed.
 * The distance between them is set as an initial translation. This mode
 * basically assumes that the anatomical objects
 * to be registered are centered in their respective images. Hence the best
 * initial guess for the registration is the one that superimposes those two
 * centers.
 *
 * In the second mode, the moments of gray level values are computed
 * for both images. The vector between the two centers of
 * mass is passes as the initial translation to the transform. This
 * second approach assumes that the moments of the anatomical objects
 * are similar for both images and hence the best initial guess for
 * registration is to superimpose both mass centers.  Note that this
 * assumption will probably not hold in multi-modality registration.
 *
 * \ingroup Transforms
 */
template< class TTransform,
class TRTransform,
class TFixedImage,
class TMovingImage >
class TranslationStackTransformInitializer : public Object
{
public:

  /** Standard class typedefs. */
  typedef TranslationStackTransformInitializer Self;
  typedef Object                          Superclass;
  typedef SmartPointer< Self >            Pointer;
  typedef SmartPointer< const Self >      ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TranslationStackTransformInitializer, Object );

  /** Type of the transform to initialize */
  typedef TTransform                                        TransformType;
  typedef typename TransformType::Pointer                   TransformPointer;
  typedef TRTransform                                       ReducedDimensionTransformType;
  typedef typename ReducedDimensionTransformType::Pointer   ReducedDimensionTransformPointer;


  /** Dimension of parameters. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, TransformType::InputSpaceDimension );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, TransformType::OutputSpaceDimension );
  itkStaticConstMacro( ReducedInputSpaceDimension, unsigned int, TransformType::InputSpaceDimension -1 );
  itkStaticConstMacro( ReducedOutputSpaceDimension, unsigned int, TransformType::OutputSpaceDimension -1 );


  /** Image Types to use in the initialization of the transform */
  typedef TFixedImage                                  FixedImageType;
  typedef TMovingImage                                 MovingImageType;
  typedef typename MovingImageType::ConstPointer       MovingImagePointer;
  typedef Image< unsigned char, OutputSpaceDimension>  MovingMaskType;
  typedef typename MovingMaskType::ConstPointer        MovingMaskPointer;

  /** Point type. */
  typedef typename TransformType::InputPointType InputPointType;
  typedef typename ReducedDimensionTransformType::InputPointType ReducedDimensionInputPointType;


  /** Vector type. */
  typedef typename TransformType::OutputVectorType OutputVectorType;
  typedef typename ReducedDimensionTransformType::OutputVectorType ReducedDimensionOutputVectorType;

  typedef itk::ImageRegionConstIteratorWithIndex< MovingImageType > IteratorType;


  /** Set the transform to be initialized */
  itkSetObjectMacro( Transform, TransformType );
    
  /** Set the moving image used in the registration process */
  itkSetConstObjectMacro( MovingImage, MovingImageType );

  /** Set the moving image mask used in the registration process */
  itkSetConstObjectMacro( MovingMask, MovingMaskType );

  /** Initialize the transform using data from the images */
  virtual void InitializeTransform() const;

protected:

  TranslationStackTransformInitializer();
  ~TranslationStackTransformInitializer(){}

  void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  TranslationStackTransformInitializer( const Self & ); // purposely not implemented
  void operator=( const Self & );                  // purposely not implemented

  TransformPointer                  m_Transform;
  MovingImagePointer                m_MovingImage;
  MovingMaskPointer                 m_MovingMask;

};

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTranslationStackTransformInitializer.hxx"
#endif

#endif /* __itkTranslationStackTransformInitializer_h */
