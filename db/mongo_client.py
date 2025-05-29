import os
import logging
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv


def conectar():
    """
    Estabelece conexão com o MongoDB usando variáveis de ambiente.
    Retorna uma instância do banco de dados ou None em caso de erro.
    """
    try:
        # Carrega variáveis de ambiente
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        
        if not mongo_uri:
            raise ValueError("MONGO_URI não encontrada nas variáveis de ambiente")
            
        # Estabelece conexão
        client = MongoClient(mongo_uri)
        
        # Seleciona o banco de dados
        db = client["chat_bot"]
        return db
    
    except Exception as e:
        print(f"Erro ao conectar ao MongoDB: {e}")
        return None
    
db = conectar()  

# Exemplo de uso das coleções
colecao_usuarios = db["usuarios"]
colecao_conversas = db["conversas"]
  
def cadastrar_usuario(nome: str, telefone: str, senha: str, email: str, nascimento: str) -> tuple:
    """
    Cadastra um novo usuário no banco de dados.
    Returns: (user_id, message) tuple
    """
    try:
        # Verifica se email já existe
        if colecao_usuarios.find_one({"email": email}):
            return None, "Email já cadastrado"
            
        # Converte string de nascimento para datetime
        try:
            ano, mes, dia = nascimento.split('/')
            data_nascimento = datetime(int(ano), int(mes), int(dia))
        except (ValueError, IndexError):
            return None, "Formato de data inválido. Use aaaa/mm/dd"
            

        usuario = {
            "nome": nome,
            "telefone": telefone,
            "email": email,
            "senha": senha,
            "nascimento": data_nascimento,
            "data_cadastro": datetime.now()
        }
        
        resultado = colecao_usuarios.insert_one(usuario)
        return resultado.inserted_id, "Usuário cadastrado com sucesso!"
        
    except Exception as e:
        return None, f"Erro ao cadastrar usuário: {str(e)}"

def login_usuario(email: str, senha: str) -> dict:
    """
    Realiza o login do usuário.
    Returns: user dict or None
    """
    try:
        usuario = colecao_usuarios.find_one({
            "email": email,
            "senha": senha  # Em produção, verificar hash
        })
        return usuario
    except Exception as e:
        print(f"Erro ao fazer login: {e}")
        return None

def get_historico_usuario(usuario_id: str, limite: int = 10) -> list:
    """
    Recupera o histórico de conversas do usuário.
    """
    try:
        # Busca as últimas conversas do usuário
        conversas = list(colecao_conversas.find(
            {"cod": usuario_id}
        ).sort("timestamp", -1).limit(limite))
        
        logging.info(f"Recuperadas {len(conversas)} conversas para o usuário {usuario_id}")
        return conversas
        
    except Exception as e:
        logging.error(f"Erro ao recuperar histórico: {e}")
        return []

def listar_pessoas():
  try:
    print("\nPessoas:")
    for pessoa in colecao_conversas.find().sort("codigo"):
      print("codigo: "+str(pessoa['codigo'])+" nome: "+pessoa['nome'])
    return True
  except Exception as e:
    print(f"Erro ao listar pessoas: {e}")
    return False

def buscar_por_codigo(codigo: int):
  try:
    pessoa = colecao_conversas.find_one({"codigo": codigo})
    if pessoa:
      print("\nPessoa encontrada:")
      print("codigo: "+str(pessoa['codigo'])+" nome: "+pessoa['nome'])
      return pessoa
    else:
      print(f"Nenhuma pessoa encontrada com código {codigo}")
      return None
  except Exception as e:
    print(f"Erro ao buscar pessoa: {e}")
    return None

def atualizar_pessoa(codigo: int, novo_nome: str):
  try:
    resultado = colecao_conversas.update_one(
    {"codigo": codigo},
    {"$set": {"nome": novo_nome}}
    )
    if resultado.modified_count > 0:
      print(f"Pessoa com código {codigo} atualizada com sucesso!")
      return True
    else:
      print(f"Nenhuma pessoa encontrada com código {codigo}")
      return False
  except Exception as e:
    print(f"Erro ao atualizar pessoa: {e}")
    return False

'''def deletar_pessoa(codigo: int):
  try:
    resultado = colecao_conversas.delete_one({"codigo": codigo})
    if resultado.deleted_count > 0:
      print(f"Pessoa com código {codigo} removida com sucesso!")
      return True
    else:
      print(f"Nenhuma pessoa encontrada com código {codigo}")
      return False
  except Exception as e:
    print(f"Erro ao deletar pessoa: {e}")
    return False

def menu():
  while True:
    print("\n--- MENU ---")
    print("1. Criar nova pessoa")
    print("2. Listar todas as pessoas")
    print("3. Buscar pessoa por código")
    print("4. Atualizar pessoa")
    print("5. Deletar pessoa")
    print("0. Sair")
    opcao = input("Escolha uma opção: ")

    if opcao == "1":
      nome = input("Digite o nome: ")
      telefone = input("Digite o telefone: ")
      email = input("Digite o email: ")

      while True:
        nascimento = input("Digite a data de nascimento (aaaa/mm/dd): ")
        try:
          ano, mes, dia = nascimento.split('/')
          nascimento = datetime(int(ano), int(mes), int(dia))
        except ValueError:
          print("Formato de data inválido. Use o formato aaaa/mm/dd.")
          continue
        except IndexError:
          print("Formato de data inválido. Use o formato aaaa/mm/dd.")
          continue
        break
      cadastrar_usuario(nome, telefone, senha, email, nascimento)

    elif opcao == "2":
      listar_pessoas()

    elif opcao == "3":
      codigo = int(input("Digite o código para buscar: "))
      buscar_por_codigo(codigo)

    elif opcao == "4":
      codigo = int(input("Digite o código da pessoa a atualizar: "))
      novo_nome = input("Digite o novo nome: ")
      atualizar_pessoa(codigo, novo_nome)

    elif opcao == "5":
      codigo = int(input("Digite o código da pessoa a deletar: "))
      deletar_pessoa(codigo)

    elif opcao == "0":
      print("Saindo do sistema...")
      break

    else:
      print("Opção inválida! Tente novamente.")'''

def conversas_usuarios(bd):
    agregacao = [
        {
            "$lookup": {
                "from": "usuarios",
                "localField": "codigo",
                "foreignField": "cod",
                "as": "usuario"
            }
        }
    ]

    return list(colecao_conversas.aggregate(agregacao))

def armazenar_conversas(bd, usuario_id, pergunta, resposta):
    conversa = {
        "cod": usuario_id,
        "mensagens": [
            {"tipo": "usuario", "texto": pergunta},
            {"tipo": "bot", "texto": resposta}
        ]
    }

    resultado = colecao_conversas.insert_one(conversa)
    return resultado.inserted_id

#if __name__ == "__main__":
  #menu()