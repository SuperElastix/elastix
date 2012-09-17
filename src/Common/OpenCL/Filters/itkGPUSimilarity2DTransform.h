/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUSimilarity2DTransform_h
#define __itkGPUSimilarity2DTransform_h

#include "itkSimilarity2DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUSimilarity2DTransform
 */
template< class TScalarType = float,
          class TParentImageFilter = Similarity2DTransform< TScalarType > >
class GPUSimilarity2DTransform :
  public TParentImageFilter,
  public GPUMatrixOffsetTransformBase< TScalarType, 2, 2 >
{
public:
  /** Standard class typedefs. */
  typedef GPUSimilarity2DTransform                          Self;
  typedef TParentImageFilter                                CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType, 2, 2 > GPUSuperclass;
  typedef SmartPointer< Self >                              Pointer;
  typedef SmartPointer< const Self >                        ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUSimilarity2DTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType &        GetCPUMatrix( void ) const { return this->GetMatrix(); }
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const { return this->GetInverseMatrix(); }
  virtual const CPUOutputVectorType &  GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUSimilarity2DTransform() {}
  virtual ~GPUSimilarity2DTransform() {}

private:
  GPUSimilarity2DTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );         // purposely not implemented
};

/** \class GPUSimilarity2DTransformFactory
* \brief Object Factory implementation for GPUSimilarity2DTransform
*/
class GPUSimilarity2DTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUSimilarity2DTransformFactory Self;
  typedef ObjectFactoryBase               Superclass;
  typedef SmartPointer< Self >            Pointer;
  typedef SmartPointer< const Self >      ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUSimilarity2DTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUSimilarity2DTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUSimilarity2DTransformFactory::Pointer factory =
      GPUSimilarity2DTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUSimilarity2DTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                  // purposely not implemented

#define OverrideSimilarity2DTransformTypeMacro( st )                   \
  {                                                                    \
    this->RegisterOverride(                                            \
      typeid( Similarity2DTransform< st > ).name(),                    \
      typeid( GPUSimilarity2DTransform< st > ).name(),                 \
      "GPU Similarity2DTransform Override",                            \
      true,                                                            \
      CreateObjectFunction< GPUSimilarity2DTransform< st > >::New() ); \
  }

  GPUSimilarity2DTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideSimilarity2DTransformTypeMacro( float );
      //OverrideSimilarity2DTransformTypeMacro( double );

      OverrideSimilarity2DTransformTypeMacro( float );
      //OverrideSimilarity2DTransformTypeMacro( double );

      OverrideSimilarity2DTransformTypeMacro( float );
      //OverrideSimilarity2DTransformTypeMacro( double );
    }
  }
};
} // end namespace itk

#endif /* __itkGPUSimilarity2DTransform_h */
